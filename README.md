# quantization_lib

`quantization_lib` is a Polars-based library for configurable data
quantization and data aggregation.

It turns raw rows into quantized signatures, groups matching signatures, and
keeps the configured aggregate values for each group. This is useful when a
dataset is too large or too detailed for downstream work, but the important
structure can be represented by binned numeric columns plus summary
aggregations.

## Installation

```bash
pip install quantization_lib
```

For local development from this repository:

```bash
pip install -e .
```

## Quick Start

```python
from datasets import load_dataset

from quantization.aggregation import AggValueCounts, AggregationPlan
from quantization.api import quantize
from quantization.config import QuantizationConfig
from quantization.quant_columns import FixedStepQuantColumnConfig

ds = load_dataset("scikit-learn/iris")
df = ds["train"].to_polars()
species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

config = QuantizationConfig(
    quant_column_configs=[
        FixedStepQuantColumnConfig("SepalLengthCm", step=0.5),
        FixedStepQuantColumnConfig("SepalWidthCm", step=0.25),
    ],
    agg_plan=AggregationPlan(
        per_column={
            "PetalLengthCm": "mean",
            "PetalWidthCm": "mean",
            # AggValueCounts only counts values listed here. Missing values are
            # ignored unless None is included; they can still be derived from
            # __count minus the counted values.
            "Species": AggValueCounts(values=species),
        },
        default_non_quant=[],
        default_quant=None,
    ),
    output_dir="examples/output",
)

quantized = quantize(df, config).sort(
    ["SepalLengthCm__quant", "SepalWidthCm__quant"]
)
```

The result contains one row per quantized sepal signature, mean petal
measurements, species value-count structs, and a `__count` column with the
number of original Iris rows represented by each signature.

## Configuration

A `QuantizationConfig` combines:

- `quant_column_configs`: finalized quantization rules for the columns that
  define the signature.
- `agg_plan`: per-column and default aggregation rules.
- `output_dir`: where streaming steps write intermediate and final parquet
  files.

Built-in quantization columns include fixed-step rounding, fixed-edge binning,
decimal rounding, and data-dependent unfinalized configs. Built-in aggregations
include `off`, `min`, `max`, `mean`, circular mean, and value counts.

## Finalized Configs

`quantize`, `merge_quantized`, and the streaming APIs expect finalized
quantization column configs. Fixed-step, fixed-edge, and decimal rounding
configs are already finalized. Data-dependent configs, such as uncertainty- or
probability-based configs, should first be resolved with
`finalize_quant_column_configs(...)`, optionally reusing
`collect_frame_statistics(...)` when you need explicit control over the
statistics pass.

## Quantize And Merge

Use `quantize(frame, config)` when the input fits comfortably in memory. If the
raw data does not fit as one dataset, split it into multiple datasets or chunks,
quantize each one with the same finalized config, and then call
`merge_quantized(frames_quant, config)` to combine the quantized outputs.

For larger workflows, use:

- `quantize_streaming(frame, config, chunk_size=...)` to write quantized parquet
  chunks to `config.output_dir`.
- `merge_streaming(frames_quant, config, chunk_size=...)` to merge many
  quantized frames hierarchically without loading every intermediate result at
  once.

## Examples

See [`examples/iris_basic.py`](examples/iris_basic.py) for the runnable Iris
example. The example intentionally does not use streaming; for small in-memory
dataframes, `quantize` is the shortest path.

## Public API

The main entry points are defined in [`src/quantization/api.py`](src/quantization/api.py).