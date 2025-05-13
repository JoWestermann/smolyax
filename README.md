# Smolyax

Fast interpolation of high-dimensional and vector-valued functions
- avoiding the curse-of-dimensionality by using sparse-grid interpolation nodes, 
- ensuring numerical stability by using a barycentric Smolyak interpolation formulation 
and 
- providing hardware-agnostic high performance by implementing key algorithms in JAX.

### Features
- Node sequences for interpolation on bounded or unbounded domains 
(Leja nodes and Gauß-Hermite nodes, respectively)
- General anisotropic multi-index sets $\Lambda \subset \mathbb{N}_{0}^{d}$ of the form 
$\Lambda := \\{\boldsymbol{\nu} \in \mathbb{N}^d_0  \ : \ \sum_{j=1}^{d} k_j \nu_j < t \\}$
where $\boldsymbol{k}\in \mathbb{R}^{d}$ is monotonically increasing and controls the anisotropy
while the threshold $t > 0$ controls the cardinality of the set.
- Smolyak operator for interpolating high-dimensional and vector-valued functions 
$f : \mathbb{R}^{d_1} \to \mathbb{R}^{d_2}$ for $d_1, d_2 \in \mathbb{N}$ potentially large. 

The implementation is designed for maximal efficiency. 
As a rough example consider interpolation a scalar function with $d_1 = 10^3$ inputs using $n = 10^4$ interpolation nodes.
With `smolyax`  you can expect to both generate the multi-index set 
as well as evaluate the interpolant from the corresponding polynomial space
in well less under $0.1$ seconds on a contemporary laptop CPU.

### Documentation

- For an introduction to relevant literature and the key implementation concepts, see the [JOSS paper draft](https://github.com/JoWestermann/smolyax/blob/main/paper/paper.md) accompanying this repository.
- For code documentation see [here](https://github.com/JoWestermann/smolyax/tree/main/docs).

## Get started

### Dependencies

```
jax
numba
```

### Installation

```
pip install git+https://github.com/JoWestermann/smolyax.git
```

### Usage

To construct the interpolant to a function `f`, which has `d_in` inputs and `d_out` outputs, 
first choose the polynomial space in which to interpolate
by setting up a node generator object, e.g. Leja nodes:
```
node_gen = nodes.Leja(dim=d_in)
```
and choosing a weight vector `k` controlling the anisotropy as well as a threshold `t` controlling the size of 
the multi-index set:
```
k = [np.log((2+j)/np.log(2)) for j in range(d_in)]
t = 5.
```
Then, initialize the interpolant as
```
f_ip = interpolation.SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, d_out=d_out, t=t, f=f)
```
and evaluate it at a point `x` by calling
```
y = f_ip(x)
```

For more examples and visualizations see [notebooks](https://github.com/JoWestermann/smolyax/tree/main/notebooks), 
in particular see the examples for interpolating a 
[one-dimensional](https://github.com/JoWestermann/smolyax/blob/main/notebooks/smolyak_example_1D.ipynb), 
[two-dimensional](https://github.com/JoWestermann/smolyax/blob/main/notebooks/smolyak_example_2D.ipynb)
or [high-dimensional function](https://github.com/JoWestermann/smolyax/blob/main/notebooks/smolyak_example_high_D.ipynb).


## Contribute

### Need a feature?
We keep track of features that could be implemented without too much trouble 
and that we will work on prioritized on demand via our 
[open issues](https://github.com/JoWestermann/smolyax/issues?q=is%3Aissue%20state%3Aopen%20label%3Aenhancement).

### Submit a feature!
If you want to submit a feature, please do so via a pull request. Ensure that all tests run through by running
```
pytest
```
from the project root directory, and ensure that performance has not degraded by first creating a benchmark on the main branch via
```
pytest --benchmark-only --benchmark-save=baseline
```
and compare performance on your feature branch against this baseline via
```
pytest --benchmark-only --benchmark-compare=0001_baseline --benchmark-sort=name --benchmark-compare-fail=min:5%
```

## Cite

If you used this library for your research, please cite [the paper]():

```
@article{westermann2025smolyax
  title={Measure transport via polynomial density surrogates},
  author={Westermann, Josephine and Chen, Joshua},
  journal={tba},
  year={2025},
  doi={tba},
  url={tba},
}
```
