import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quantization.aggregation import AggregationPlan
from quantization.api import collect_frame_statistics, merge_quantized, merge_streaming, quantize_streaming
from quantization.config import QuantizationConfig
from quantization.quant_columns import FixedStepQuantColumnConfig


@pytest.mark.parametrize("as_lazy", [False, True])
def test_quantize_streaming(tmp_path, as_lazy):
    df = pl.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    frame = df.lazy() if as_lazy else df

    config = QuantizationConfig(
        quant_column_configs=[FixedStepQuantColumnConfig(name="val", step=2.0)],
        agg_plan=AggregationPlan(per_column={"val": "off"}, default_non_quant=[]),
        output_dir=tmp_path,
    )

    quantize_streaming(frame, config, chunk_size=2)

    parquet_files = list(sorted(tmp_path.glob("*.parquet")))
    assert len(parquet_files) == 5

    quantized_df = pl.read_parquet(parquet_files[2]).sort("val__quant", descending=True)
    expected = pl.DataFrame(
        {
            "val__quant": [6.0, 4.0],
            "val__off": [[6], [5]],
            "__count": pl.Series([1, 1], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(quantized_df, expected)


@pytest.mark.parametrize("as_lazy", [False, True])
def test_merge_streaming(tmp_path, as_lazy):
    chunk_size = 1

    frames_df = [
        pl.DataFrame({"val__quant": [2.0, 4.0, 6.0, 8.0], "__count": [1, 1, 1, 1]}),
        pl.DataFrame({"val__quant": [2.0, 6.0, 10.0, 12.0], "__count": [1, 1, 1, 1]}),
        pl.DataFrame({"val__quant": [4.0, 8.0, 10.0, 14.0], "__count": [1, 1, 1, 1]}),
        pl.DataFrame({"val__quant": [2.0, 8.0, 12.0, 14.0], "__count": [1, 1, 1, 1]}),
        pl.DataFrame({"val__quant": [6.0, 8.0, 12.0, 16.0], "__count": [1, 1, 1, 1]}),
    ]
    frames = [df.lazy() for df in frames_df] if as_lazy else frames_df

    config = QuantizationConfig(
        quant_column_configs=[FixedStepQuantColumnConfig(name="val", step=2.0)],
        agg_plan=AggregationPlan(per_column={}, default_non_quant=[]),
        output_dir=tmp_path,
    )

    merged = merge_streaming(frames, config, chunk_size=chunk_size).sort("val__quant")

    actual_tree = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.parquet"))
    expected_tree = sorted(
        [
            "level_0/merged_0.parquet",
            "level_0/merged_1.parquet",
            "level_0/merged_2.parquet",
            "level_1/merged_0.parquet",
            "level_1/merged_1.parquet",
            "merged_final.parquet",
        ]
    )
    assert actual_tree == expected_tree

    expected = merge_quantized(frames_df, config).sort("val__quant")
    assert_frame_equal(merged, expected)


def _double_x(chunk: pl.DataFrame) -> pl.DataFrame:
    return chunk.select((pl.col("x") * 2).alias("val"))


@pytest.mark.parametrize("as_lazy", [False, True])
def test_quantize_streaming_with_preprocess(tmp_path, as_lazy):
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    frame = df.lazy() if as_lazy else df

    def get_config(output_dir):
        return QuantizationConfig(
            quant_column_configs=[FixedStepQuantColumnConfig(name="val", step=2.0)],
            agg_plan=AggregationPlan(per_column={"val": "off"}, default_non_quant=[]),
            output_dir=output_dir,
        )

    quantize_streaming(frame, get_config(tmp_path / "preprocess"), chunk_size=2, preprocess=_double_x)
    quantize_streaming(_double_x(df), get_config(tmp_path / "expected"), chunk_size=2)

    actual_files = sorted((tmp_path / "preprocess").glob("*.parquet"))
    expected_files = sorted((tmp_path / "expected").glob("*.parquet"))
    assert [p.name for p in actual_files] == [p.name for p in expected_files]
    for actual, expected in zip(actual_files, expected_files):
        assert_frame_equal(pl.read_parquet(actual).sort("val__quant"), pl.read_parquet(expected).sort("val__quant"))


@pytest.mark.parametrize("as_lazy", [False, True])
def test_collect_frame_statistics_with_preprocess(as_lazy):
    df = pl.DataFrame({"x": [float(i) / 3 for i in range(100)]})
    frame = df.lazy() if as_lazy else df

    kwargs = dict(quant_columns=["val"], uncertainties={"val": 1.0}, factors=[1.0, 2.0], chunk_size=10)
    actual = collect_frame_statistics(frame, preprocess=_double_x, **kwargs)
    expected = collect_frame_statistics(_double_x(df), **kwargs)

    assert_frame_equal(actual.column2ecdf["val"], expected.column2ecdf["val"])
    assert actual.factor2size == expected.factor2size
