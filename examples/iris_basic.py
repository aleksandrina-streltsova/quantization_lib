from datasets import load_dataset

from quantization.aggregation import AggValueCounts, AggregationPlan
from quantization.api import quantize
from quantization.config import QuantizationConfig
from quantization.quant_columns import FixedStepQuantColumnConfig


def main():
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

    print(f"Input rows: {df.height}")
    print(f"Quantized rows: {quantized.height}")
    print(quantized.head(10))


if __name__ == "__main__":
    main()
