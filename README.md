# BottleneckTest

[![CI](https://github.com/kapildhinwa2010-dev/BottleneckTest/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/BottleneckTest/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/BottleneckTest/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/BottleneckTest/actions/workflows/codeql.yml)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Detect low-rank bottlenecks in stacked weight layers via singular-value spectra.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

Given a dict of 2-D weight matrices, analyze_model() reports per layer:
singular spectrum, effective rank (smallest k covering 99% of spectral
energy), condition number (inf when numerically singular), and norms.
find_bottlenecks() ranks layers by smallest effective rank. This is the
canonical spectral-base module of the toolkit; DistributionShift
extends it with between-sets spectrum comparison.

## Install

    pip install -r requirements.txt

## Quick start

    from bottlenecktest import analyze_model, find_bottlenecks

    reports = analyze_model({"a": np.eye(16), "b": low_rank})
    print(find_bottlenecks(reports, top_k=1))

## API

- layer_spectrum(weights) - singular values, descending
- effective_rank(weights, energy=0.99)
- layer_report(weights) - shape, norms, condition_number, effective_rank
- analyze_model(layers) / find_bottlenecks(reports, top_k=1)

## Honest results and limitations

- "Bottleneck" here means low effective rank of the weight matrix,
  which correlates with - but does not prove - a training bottleneck.
  No accuracy or gradient measurement is included.
- Effective rank uses a fixed 99% energy cutoff; spectra with slow
  decay get high ranks even when visibly compressible.
- condition_number is inf for numerically singular matrices rather
  than a huge finite number; handle it in downstream code.
- Full SVD per layer: O(min(m,n)^2 * max(m,n)) time. Fine for toy
  scales, slow for very large layers.

## License

MIT - see [LICENSE](LICENSE).
