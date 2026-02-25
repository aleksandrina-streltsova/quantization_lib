import pytest

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
