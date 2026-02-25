import pytest
import polars as pl
from quantization.aggregation import AggOff
from tests.aggregation.utils import run_quantize_test, run_merge_test

@pytest.fixture
def agg_off():
    return AggOff()

def test_off_quantize(agg_off):
    input_data = {"val": [1, 2, 3, None]}
    expected_data = {"val__off": [[1, 2, 3, None]]}
    run_quantize_test(agg_off, input_data, expected_data)

def test_off_merge(agg_off):
    # AggOff merge explodes the list then agg collects it back
    quantized_data = {"val__off": [[1, 2, None], [3]]}
    expected_data = {"val__off": [[1, 2, None, 3]]}
    run_merge_test(agg_off, quantized_data, expected_data)
