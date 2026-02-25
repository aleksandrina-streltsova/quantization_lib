import pytest
import polars as pl
from quantization.aggregation import AggValueCounts
from tests.aggregation.utils import run_quantize_test, run_merge_test

@pytest.fixture
def agg_vc():
    return AggValueCounts(values=[1, 2, 3, None])

def test_value_counts_quantize(agg_vc):
    input_data = {"val": [1, 1, 2, 4, None]}
    # 4 is not in [1, 2, 3], but quantize includes all values present.
    # value_counts() returns all values present.

    expected_data = {
        "val__valuecounts": [[
            {"val": 1, "struct_count": 2},
            {"val": 2, "struct_count": 1},
            {"val": 4, "struct_count": 1},
            {"val": None, "struct_count": 1},
        ]]
    }
    # Note: assert_frame_equal() handles the issue with the order in value_counts being non-deterministic.
    run_quantize_test(agg_vc, input_data, expected_data)

def test_value_counts_merge(agg_vc):
    # Merge only for values specified in AggValueCounts(values=[1, 2, 3])
    quantized_data = {
        "val__valuecounts": [
            [{"val": 1, "struct_count": 2}, {"val": 2, "struct_count": 1}, {"val": 4, "struct_count": 1},
             {"val": None, "struct_count": 1}],
            [{"val": 1, "struct_count": 3}, {"val": 3, "struct_count": 1},
             {"val": None, "struct_count": 1}],
        ]
    }
    expected_data = {
        "val__valuecounts": [[
            {"val": 1, "struct_count": 5},
            {"val": 2, "struct_count": 1},
            {"val": 3, "struct_count": 1},
            {"val": None, "struct_count": 2},
        ]]
    }
    run_merge_test(agg_vc, quantized_data, expected_data)
