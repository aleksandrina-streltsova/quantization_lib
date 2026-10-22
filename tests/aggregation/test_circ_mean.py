import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quantization.aggregation import AggCircularMean
from tests.aggregation.utils import run_quantize_test, run_merge_test


@pytest.fixture
def agg_circ():
    # Test with hours (0, 24)
    return AggCircularMean(low=0, high=24)


def test_circular_mean_quantize(agg_circ):
    # 23:00 and 01:00 should have a mean of 0.0 or 24.0
    input_data = {"time": [23.0, 1.0, None]}

    expected_data = {
        "time__circmean": [0.0],
        "time__circmeancount": [2],
        "time__circmeanr": [0.9659258262890682],
    }
    run_quantize_test(agg_circ, input_data, expected_data, column="time")


def test_circular_mean_merge(agg_circ):
    quantized_data = {
        "time__circmean": [18.0, 6.0, None],
        "time__circmeancount": [1, 2, 0],
        "time__circmeanr": [1.0, 1.0, None],
    }
    expected_data = {
        "time__circmean": [6.0],
        "time__circmeancount": [3],
        "time__circmeanr": [1 / 3],
    }
    run_merge_test(agg_circ, quantized_data, expected_data, column="time")


def test_circular_mean_lon():
    # Test with longitude (-180, 180)
    agg = AggCircularMean(low=-180, high=180)

    # 170 and -170 should have a mean of 180 (or -180)
    input_data = {"lon": [170.0, -170.0, None]}

    expected_data = {
        "lon__circmean": [180.0],
        "lon__circmeancount": [2],
        "lon__circmeanr": [0.9848077530122079],
    }
    run_quantize_test(agg, input_data, expected_data, column="lon")


def test_circular_mean_merge_matches_single_pass(agg_circ):
    # Merging partial means must give the single-pass result, whatever the split
    values = [23.0, 1.0, 2.0, 5.0, 11.0, 12.0, 18.0]
    single_pass = pl.DataFrame({"time": values}).select(agg_circ.expr_quantize("time"))
    for split in range(1, len(values)):
        partial = pl.concat([pl.DataFrame({"time": values[:split]}).select(agg_circ.expr_quantize("time")),
                             pl.DataFrame({"time": values[split:]}).select(agg_circ.expr_quantize("time"))])
        merged = partial.select(agg_circ.expr_merge("time"))
        assert_frame_equal(merged, single_pass, check_dtypes=False)


def test_circular_mean_merge_all_null(agg_circ):
    # Parts with no values give a null mean and length, not low and 0 / 0 = NaN
    quantized_data = {
        "time__circmean": pl.Series([None, None], dtype=pl.Float64),
        "time__circmeancount": [0, 0],
        "time__circmeanr": pl.Series([None, None], dtype=pl.Float64),
    }
    expected_data = {
        "time__circmean": pl.Series([None], dtype=pl.Float64),
        "time__circmeancount": [0],
        "time__circmeanr": pl.Series([None], dtype=pl.Float64),
    }
    run_merge_test(agg_circ, quantized_data, expected_data, column="time")


def test_circular_mean_merge_twice_with_all_null_part(agg_circ):
    # A merged all-null part must not turn later merges into NaN
    all_null = pl.DataFrame({"time": pl.Series([None, None], dtype=pl.Float64)}).select(agg_circ.expr_quantize("time"))
    merged_all_null = pl.concat([all_null, all_null]).select(agg_circ.expr_merge("time"))
    valued = pl.DataFrame({"time": [23.0, 1.0, 2.0]}).select(agg_circ.expr_quantize("time"))
    merged = pl.concat([merged_all_null, valued], how="vertical_relaxed").select(agg_circ.expr_merge("time"))
    assert_frame_equal(merged, valued, check_dtypes=False)
