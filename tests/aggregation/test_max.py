import pytest
from quantization.aggregation import AggMax
from tests.aggregation.utils import run_quantize_test, run_merge_test

@pytest.fixture
def agg_max():
    return AggMax()

def test_max_quantize(agg_max):
    input_data = {"val": [10, 2, 30, None]}
    expected_data = {"val__max": [30]}
    run_quantize_test(agg_max, input_data, expected_data)

def test_max_merge(agg_max):
    quantized_data = {"val__max": [2, 5, 1, None]}
    expected_data = {"val__max": [5]}
    run_merge_test(agg_max, quantized_data, expected_data)
