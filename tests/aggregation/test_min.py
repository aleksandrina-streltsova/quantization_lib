import pytest
from quantization.aggregation import AggMin
from tests.aggregation.utils import run_quantize_test, run_merge_test

@pytest.fixture
def agg_min():
    return AggMin()

def test_min_quantize(agg_min):
    input_data = {"val": [10, 2, 30, None]}
    expected_data = {"val__min": [2]}
    run_quantize_test(agg_min, input_data, expected_data)

def test_min_merge(agg_min):
    quantized_data = {"val__min": [2, 5, 1, None]}
    expected_data = {"val__min": [1]}
    run_merge_test(agg_min, quantized_data, expected_data)
