# ----------------------------------------------------------------------------
# Installing TASMANIAN
#   * export CMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc
#   * export Tasmanian_ENABLE_CUDA=/usr/local/cuda-12.2/bin/nvcc
#   * pip install --no-cache-dir Tasmanian --no-binary=Tasmanian Tasmanian
# ----------------------------------------------------------------------------
import argparse
import GPUtil
import os
import socket
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import jax
jax.config.update("jax_platform_name", "gpu")

import Tasmanian
from smolyax import indices, nodes
from smolyax.interpolation import SmolyakBarycentricInterpolator

import testfunction, depthmap


# ----------------------------------------------------------------------------
# USER PARAMETERS
# ----------------------------------------------------------------------------


TARGET_N_LIST = [1_000, 2_000, 4_000, 6_000, 8_000, 10_000]
N_MC_LIST     = [50, 500, 5000]
D_IN_LIST     = [10, 40, 160]
D_OUT_LIST    = [10, 40, 160]

N_ITER = 20

SMOLYAX_MEMORY_LIMIT = 20.  # Adjust to GPU capabilities (leave a little bit space for overhead)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

HOST = socket.gethostname()
FLAG = '_1'

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------


def tasmanian_filename(d_in: int, d_out: int, depth: int, n_mc: int) -> Path:
    return RESULTS_DIR / (
        f"tasmanian_din{d_in}_dout{d_out}_depth{depth}_nmc{n_mc}_niter{N_ITER}_{HOST}.npz"
    )


def smolyax_filename(d_in: int, d_out: int, N_target: int, n_mc: int) -> Path:
    return RESULTS_DIR / (
        f"smolyax_din{d_in}_dout{d_out}_N{N_target}_nmc{n_mc}_niter{N_ITER}_{HOST}_{SMOLYAX_MEMORY_LIMIT}{FLAG}.npz"
    )


def rmse(y_true, y_pred) :
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2))


# ----------------------------------------------------------------------------
# BENCHMARK FUNCTIONS
# ----------------------------------------------------------------------------


def run_tasmanian(
    d_in: int,
    d_out: int,
    depth: int,
    n_mc: int,
    f_test,
    aniso,
    X_test,
    Y_test,
    use_saved=True,
):
    fname = tasmanian_filename(d_in, d_out, depth, n_mc)
    if fname.is_file() and use_saved:
        d = np.load(fname)
        return float(d["e"][()]), int(d["N"][()]), float(d["t"][()])

    grid = Tasmanian.makeSequenceGrid(d_in, d_out, depth, "hyperbolic", "leja", aniso)
    grid.enableAcceleration("gpu-cublas", 0)  # other available GPU backends do not work for d_in >= 32
    assert grid.isCudaEnabled()

    Tasmanian.loadNeededValues(f_test, grid)

    t0 = time.perf_counter()
    for _ in range(N_ITER) :
        Y = np.asarray(grid.evaluateBatch(X_test))
    runtime = time.perf_counter() - t0

    error = rmse(Y_test, Y)
    np.savez_compressed(fname, N=grid.getNumLoaded(), t=runtime, e=error)
    return error, grid.getNumLoaded(), runtime


def run_smolyax(
    d_in: int,
    d_out: int,
    N_target: int,
    n_mc: int,
    f_test,
    aniso,
    X_test,
    Y_test,
    node_gen: nodes.Generator,
    use_saved=True,
):
    fname = smolyax_filename(d_in, d_out, N_target, n_mc)
    if fname.is_file() and use_saved:
        d = np.load(fname)
        return float(d["e"][()]), int(d["N"][()]), float(d["t"][()])

    k = np.squeeze(np.array(aniso, dtype=float))
    t_val = indices.find_approximate_threshold(k, N_target, node_gen.is_nested)

    interp = SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t_val, d_out=d_out, f=f_test,
            n_inputs=n_mc, memory_limit=SMOLYAX_MEMORY_LIMIT)

    t0 = time.perf_counter()
    for _ in range(N_ITER) :
        Y = np.asarray(interp(X_test))
    runtime = time.perf_counter() - t0

    error = rmse(Y_test, Y)
    np.savez_compressed(fname, N=interp.n_f_evals, t=runtime, e=error)
    return error, interp.n_f_evals, runtime


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    # -- setup GPUS ---
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print("\t", gpu.id, gpu.memoryFree, gpu.load)
    filtered_gpus = [g.id for g in sorted([g for g in gpus if g.memoryFree >= 10_000 and g.load < 0.5], key=lambda g: g.load)]
    if filtered_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, filtered_gpus))
        print("CUDA_VISIBLE_DEVICES set to:", filtered_gpus)
    else:
        raise RuntimeError("No suitable GPUs available.")

    parser = argparse.ArgumentParser(description="Heat‑map benchmark Tasmanian vs. Smolyax")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for Monte‑Carlo cloud")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    log_ratio = -10 * np.zeros(
        (
            len(D_IN_LIST),
            len(D_OUT_LIST),
            len(N_MC_LIST),
            len(TARGET_N_LIST),
        ),
        dtype=float,
    )

    for i_di, d_in in enumerate(D_IN_LIST):

        print(f'D_in = {d_in}')
        f_dummy, r_dummy, theta_dummy = testfunction.build_test_function(d_in, 1)
        aniso_t = testfunction.anisotropy_weights(d_in, r_dummy[0], theta_dummy[0])
        aniso_t = list(np.maximum(1.0, np.ceil(1000 * np.log(((np.arange(d_in) + 2) ** r0) / theta0))).astype(int))
        aniso_s = [np.log(((j + 2) ** r_dummy) / theta_dummy) for j in range(d_in)]
        depth_for_N = depthmap.get_depth_map(d_in, TARGET_N_LIST, aniso_t)

        node_gen = nodes.Leja(dim=d_in)

        for j_do, d_out in enumerate(D_OUT_LIST):

            print(f'\tD_out = {d_out}')
            f_test, r, theta = testfunction.build_test_function(d_in, d_out)

            X_full = rng.uniform(-1.0, 1.0, size=(max(N_MC_LIST), d_in)).astype(np.float64, order="C")
            Y_full = f_test(X_full)

            # tasmanian warmup call
            _ = run_tasmanian(d_in, d_out, depth_for_N[TARGET_N_LIST[-1]][0], N_MC_LIST[-1], f_test, aniso_t,
                              X_full, Y_full, use_saved=False)

            for k_N, N_target in enumerate(TARGET_N_LIST):
                print(f'\t\tN = {N_target}')
                depth_sel, _ = depth_for_N[N_target]
                for l_nmc, n_mc in enumerate(N_MC_LIST):
                    print(f'\t\tn_mc = {n_mc}')
                    X_sub = X_full[:n_mc]

                    try :
                        e_tas, N_tas, t_tas = run_tasmanian(d_in, d_out, depth_sel, n_mc, f_test, aniso_t, X_sub, Y_full[:n_mc])
                        print(f'\t\t\tTAS: E={e_tas:.2e}, N={N_tas}, T=={t_tas:.3f}')
                        e_smo, N_smo, t_smo = run_smolyax(d_in, d_out, N_target, n_mc, f_test, aniso_s, X_sub, Y_full[:n_mc], node_gen)
                        print(f'\t\t\tSMO: E={e_smo:.2e}, N={N_smo}, T=={t_smo:.3f}', end=' >> ')

                        log_ratio[i_di, j_do, l_nmc, k_N] = np.log10(t_tas / t_smo)
                        print(f"log10={log_ratio[i_di,j_do,l_nmc,k_N]:+.3f}")

                    except Exception as e:
                        print(f"An error occurred: {e}")

    # --- Plot ---
    max_abs = np.nanmax(np.abs(log_ratio))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, axes = plt.subplots(len(D_IN_LIST), len(D_OUT_LIST),
                             figsize=(4 * len(D_OUT_LIST), 3 * len(D_IN_LIST)),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)

    for i_di, d_in in enumerate(D_IN_LIST):
        for j_do, d_out in enumerate(D_OUT_LIST):
            ax = axes[len(D_IN_LIST) - 1 - i_di, j_do]
            im = ax.imshow(log_ratio[i_di, j_do], origin="lower", aspect="auto", cmap="coolwarm", norm=norm)

            ax.set_xticks(range(len(TARGET_N_LIST)))
            ax.set_yticks(range(len(N_MC_LIST)))

            if i_di == 0:  # show x labels
                ax.set_xticklabels([f"{n:,}".replace(",", " ") for n in TARGET_N_LIST], rotation=45)
                ax.set_xlabel(r"# of target evaluations $|\Lambda|$")
            else:
                ax.tick_params(labelbottom=False)

            if (j_do == 0):  # show y labels
                ax.set_yticklabels([str(n) for n in N_MC_LIST])
                ax.set_ylabel(r"$n_{\rm batch}$")
            else:
                ax.tick_params(labelleft=False)

            ax.set_title(r'$d_{\rm in}=$' + rf'${d_in}$, ' + r'$d_{\rm out}=$' + rf'${d_out}$')

    cbar = fig.colorbar(im, ax=axes, shrink=1., pad=0.02)
    cbar.set_label(r"$\log_{10}(t_{\mathrm{Tas}} / t_{\mathrm{Smol}})$")

    out_png = RESULTS_DIR / f"tas_vs_smolyax_heatmap_{HOST}_{SMOLYAX_MEMORY_LIMIT}{FLAG}.png"
    fig.savefig(out_png, dpi=300)
    print("Figure saved to", out_png)
