# soccernetpy

Python bindings for two text-classification models provided by
[`soccer-rs`](https://github.com/danielruss/soccer-rs):

- **SOCcerNET** classifies job titles and tasks into US SOC 2010 occupations.
- **CLIPS** classifies product and service descriptions into NAICS 2022
  industries.

The computational code is written in Rust and exposed to Python with PyO3.
Both the distribution and its import module are named `soccernetpy`.

## Installation

Once published on PyPI:

```bash
pip install soccernetpy
```

For local development, create or activate a virtual environment and run:

```bash
maturin develop
```

To build an installable wheel:

```bash
maturin build --release
pip install ../target/wheels/soccernetpy-*.whl
```

The exact wheel filename contains its supported Python version, operating
system, and CPU architecture. A wheel built for one platform may not install on
another platform.

The first non-empty classification may download the embedding and classifier
model files. Later calls use the cached copies.

## Platform support

Prebuilt wheels are not currently available for Intel Macs
(`x86_64-apple-darwin`). Microsoft has dropped support for ONNX Runtime on
Intel macOS, and SOCcerNET requires ONNX Runtime for model inference. As a
result, we cannot support Intel Macs until a suitable workaround is available.

Apple Silicon Macs (`aarch64-apple-darwin`) remain supported.

## SOCcerNET

```python
from soccernetpy import soccernet

results = soccernet(
    job_titles=["soccer player", "coach"],
    job_tasks=["play soccer", "coach players"],
    n=3,
)

for job_candidates in results:
    for candidate in job_candidates:
        print(candidate.code, candidate.title, candidate.score)
```

Signature:

```python
soccernet(
    job_titles,
    job_tasks,
    soc1980=None,
    isco1988=None,
    noc2011=None,
    n=10,
)
```

`job_titles` and `job_tasks` must have equal lengths. Each optional prior-code
list must also contain one entry per job. A prior entry may be:

- one code, such as `"261"`;
- multiple codes, such as `["211", "212"]`; or
- `None` when that job has no prior code.

For example:

```python
results = soccernet(
    job_titles=["doctor", "lawyer", "coach"],
    job_tasks=["treat patients", "give legal advice", "train athletes"],
    soc1980=["261", ["211", "212"], None],
    n=5,
)
```

SOC 1980, ISCO 1988, and NOC 2011 priors are crosswalked into SOC 2010 before
inference. Codes not found in a crosswalk do not contribute a prior.

## CLIPS

```python
from soccernetpy import clips

results = clips(
    products_services=[
        "Software development",
        "Full-service dental care",
    ],
    sic1987=["7372", "8021"],
    n=3,
)
```

Signature:

```python
clips(products_services, sic1987=None, n=10)
```

`sic1987` follows the same one-code, multiple-code, or `None` convention
as the SOCcerNET priors. SIC 1987 priors are crosswalked into NAICS 2022.

## Results

Both functions return `list[list[SoccerResult]]`:

- The outer list has one entry per input row and preserves input order.
- Each inner list contains at most `n` ranked candidates.
- Candidates expose `code`, `title`, and `score` attributes.

```python
best = results[0][0]
print(best.code)
print(best.title)
print(best.score)
```

`n=0` returns an empty candidate list for each input. Empty input lists return
an empty outer list without loading a model.

## Errors

- Mismatched input lengths raise `ValueError`.
- A negative `n` raises `OverflowError` during Python-to-Rust conversion.
- Model, inference, cache, and crosswalk failures raise `RuntimeError`.

## Development checks

```bash
cargo fmt --check
cargo test
maturin develop
python working/test2.py
```
