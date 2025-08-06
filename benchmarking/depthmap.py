from pathlib import Path
from typing import Sequence, Dict, Tuple

import numpy as np

import Tasmanian

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def filename(d_in: int) -> Path:
    return RESULTS_DIR / f"depthmap_din{d_in}.npz"


def num_points_for_depth(d_in: int, depth: int, aniso: Sequence[int]) -> int:
    grid = Tasmanian.makeSequenceGrid(d_in, 1, depth, "hyperbolic", "leja", aniso)
    return grid.getNumPoints()


def find_depth_for_target_points(
    d_in: int,
    aniso: Sequence[int],
    target_n: int,
    max_iter: int = 32,
    rel_tol: float = 0.001,
    verbose = False,
) -> Tuple[int, int]:
    """Binary‑search integer depth achieving ≈ *target_n* nodes."""
    depth_lo, n_lo = 1, num_points_for_depth(d_in, 1, aniso)
    if n_lo >= target_n:
        return depth_lo, n_lo

    depth_hi, n_hi = depth_lo, n_lo
    while n_hi < target_n:
        depth_hi *= 2
        n_hi = num_points_for_depth(d_in, depth_hi, aniso)
    if verbose :
        print('Target:', target_n, '/ Testrange', depth_lo, '-', depth_hi)

    for i in range(max_iter):
        if depth_lo + 1 >= depth_hi:
            break
        depth_mid = (depth_lo + depth_hi) // 2
        n_mid = num_points_for_depth(d_in, depth_mid, aniso)
        if n_mid < target_n:
            depth_lo, n_lo = depth_mid, n_mid
        else:
            depth_hi, n_hi = depth_mid, n_mid
        if verbose : print(i, depth_mid, n_mid)
        if abs(n_mid - target_n) / target_n < rel_tol:
            return depth_mid, n_mid

    return (depth_lo, n_lo) if abs(n_lo - target_n) < abs(n_hi - target_n) else (depth_hi, n_hi)


def get_depth_map(d_in, TARGET_N_LIST, aniso, verbose=True) :

    depthmap_file = filename(d_in)
    depth_for_N: Dict[int, Tuple[int, int]] = {}
    if depthmap_file.is_file():
        d = np.load(depthmap_file)
        for N, dep, ach in zip(d["target"], d["depth"], d["achieved"]):
            depth_for_N[int(N)] = (int(dep), int(ach))

    for N_target in TARGET_N_LIST:
        if N_target not in depth_for_N:
            depth_sel, N_ach = find_depth_for_target_points(d_in, aniso, N_target, verbose=verbose)
            depth_for_N[N_target] = (depth_sel, N_ach)
            print(f"d_in={d_in:3d} N={N_target:7d} -> depth={depth_sel:3d}  N_ach={N_ach}")

    np.savez_compressed(
        depthmap_file,
        target=np.array(list(depth_for_N.keys()), dtype=int),
        depth=np.array([v[0] for v in depth_for_N.values()], dtype=int),
        achieved=np.array([v[1] for v in depth_for_N.values()], dtype=int),
    )

    return depth_for_N
