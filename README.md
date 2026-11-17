# BottleneckTest

BottleneckTest is a small Python library for inspecting the singular-value structure of two-dimensional weight matrices. It reports basic matrix statistics and ranks named matrices by effective rank as a simple low-rank bottleneck heuristic. It does not load or inspect framework model objects, and its ranking alone does not establish a training or accuracy bottleneck.

The project metadata identifies this as version `0.1.0`. It currently consists of a single Python module and a focused test suite; treat it as a lightweight analysis utility rather than a complete model-analysis toolkit.

## Requirements and installation

The package requires Python 3.9 or newer and NumPy 1.24 or newer. From a checkout, install it with:

```bash
python -m pip install .
```

To install the declared runtime and test dependencies directly instead, use:

```bash
python -m pip install -r requirements.txt
```

The latter installs NumPy and pytest. The project is packaged from `pyproject.toml` as `bottlenecktest`.

## Usage

Pass a mapping from layer names to two-dimensional NumPy-compatible arrays:

```python
import numpy as np
from bottlenecktest import analyze_model, find_bottlenecks

layers = {
    "projection": np.eye(4),
    "rank_one": np.ones((4, 4)),
}
reports = analyze_model(layers)
print(reports["projection"])
print(find_bottlenecks(reports, top_k=1))
```

`analyze_model` returns one report per input name. Each report contains the matrix `shape`, `frobenius_norm`, `spectral_norm`, `condition_number`, and `effective_rank`. `find_bottlenecks` returns layer names ordered by ascending reported effective rank (ties are ordered by name).

## API

- `layer_spectrum(weights)` converts its input to `float64` and returns its singular values in descending order. The input must be two-dimensional.
- `effective_rank(weights, energy=0.99)` returns the smallest number of leading singular values whose cumulative **sum** reaches the specified fraction of the sum of all singular values. The parameter must be in `(0, 1]`. Despite the parameter name, this is not a squared-singular-value or Frobenius-energy threshold. A zero matrix returns rank `1`.
- `layer_report(weights)` returns the shape, Frobenius norm, spectral norm, condition number, and default effective rank for a non-empty two-dimensional matrix. The condition number is reported as infinity when the smallest singular value is at or below a scale-dependent machine-precision threshold.
- `analyze_model(layers)` applies `layer_report` to each value in a non-empty mapping of names to matrices.
- `find_bottlenecks(reports, top_k=1)` returns up to `top_k` names with the smallest effective ranks. Empty reports produce an empty list; `top_k` must be at least one for non-empty reports.

## Tests and CI

On the checked-out repository commit, the local test suite completed with **11 passed** on Python 3.12.3. A separate API smoke test also passed. A wheel build from a temporary copy of the source and project metadata succeeded. These are local checks; no hosted GitHub Actions run or result is asserted here.

The CI workflow is configured to run pytest, compile the module and tests, and perform a small import/report smoke check on Python 3.10, 3.11, and 3.12. A separate CodeQL workflow is configured for Python analysis; its configured presence is not evidence of a particular analysis result.

Run the test suite locally with:

```bash
python -m pytest -q
```

## Scope and limitations

- The input is a mapping of matrix names to arrays; there is no integration for loading neural-network checkpoints or extracting weights from a framework model.
- “Bottleneck” means only a low effective rank under this module's singular-value sum cutoff. The module does not measure gradients, training dynamics, downstream accuracy, or causal impact on model performance.
- The effective-rank default is a fixed 99% cumulative singular-value-sum threshold. Different choices of threshold can change rankings, and this metric is not the conventional squared-spectrum energy fraction.
- Reports require non-empty two-dimensional matrices. The implementation computes an SVD for each matrix, which may be expensive for large layers; no performance benchmark or scale claim is provided.
- The repository contains unit tests, but no benchmark dataset, measured performance results, or empirical research findings.

## License

The repository includes an MIT License, copyright (c) 2026 Kapildhinwa2010-dev. See [LICENSE](LICENSE) for the complete terms.
