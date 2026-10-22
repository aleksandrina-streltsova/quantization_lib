import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quantization.aggregation import AggMean
from tests.aggregation.utils import run_quantize_test, run_merge_test


@pytest.fixture
def agg_mean():
    return AggMean()


def test_mean_quantize(agg_mean):
    input_data = {"val": [1.0, 2.0, 6.0, None]}
    expected_data = {
        "val__mean": [3.0],
        "val__meancount": [3]
    }
    run_quantize_test(agg_mean, input_data, expected_data)


def test_mean_merge(agg_mean):
    quantized_data = {
        "val__mean": [2.0, 4.0, None],
        "val__meancount": [1, 3, 0]
    }
    # (2*1 + 4*3) / (1+3) = (2 + 12) / 4 = 14 / 4 = 3.5
    expected_data = {
        "val__mean": [3.5],
        "val__meancount": [4]
    }
    run_merge_test(agg_mean, quantized_data, expected_data)


def test_mean_merge_all_null(agg_mean):
    # Parts with no values give a null mean, not 0 / 0 = NaN
    quantized_data = {
        "val__mean": pl.Series([None, None], dtype=pl.Float64),
        "val__meancount": [0, 0]
    }
    expected_data = {
        "val__mean": pl.Series([None], dtype=pl.Float64),
        "val__meancount": [0]
    }
    run_merge_test(agg_mean, quantized_data, expected_data)


def test_mean_merge_twice_with_all_null_part(agg_mean):
    # A merged all-null part must not turn later merges into NaN
    all_null = pl.DataFrame({"val": pl.Series([None, None], dtype=pl.Float64)}).select(agg_mean.expr_quantize("val"))
    merged_all_null = pl.concat([all_null, all_null]).select(agg_mean.expr_merge("val"))
    valued = pl.DataFrame({"val": [2.0, 4.0]}).select(agg_mean.expr_quantize("val"))
    merged = pl.concat([merged_all_null, valued], how="vertical_relaxed").select(agg_mean.expr_merge("val"))
    assert_frame_equal(merged, valued, check_dtypes=False)
