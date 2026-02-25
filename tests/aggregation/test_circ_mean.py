import pytest

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
        "time__circmeancount": [2]
    }
    run_quantize_test(agg_circ, input_data, expected_data, column="time")


def test_circular_mean_merge(agg_circ):
    quantized_data = {
        "time__circmean": [18.0, 6.0, None],
        "time__circmeancount": [1, 2, 0]
    }
    expected_data = {
        "time__circmean": [6.0],
        "time__circmeancount": [3]
    }
    run_merge_test(agg_circ, quantized_data, expected_data, column="time")


def test_circular_mean_lon():
    # Test with longitude (-180, 180)
    agg = AggCircularMean(low=-180, high=180)

    # 170 and -170 should have a mean of 180 (or -180)
    input_data = {"lon": [170.0, -170.0, None]}

    expected_data = {
        "lon__circmean": [180.0],
        "lon__circmeancount": [2]
    }
    run_quantize_test(agg, input_data, expected_data, column="lon")
