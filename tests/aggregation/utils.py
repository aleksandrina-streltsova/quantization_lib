from typing import Any

import polars as pl
from quantization.aggregation import Aggregation
from quantization.types import Frame

from tests.utils import run_test_twice


def run_quantize_test(agg: Aggregation, input_data: dict[str, Any], expected_data: dict[str, Any], column: str = "val"):
    _run_test(agg.expr_quantize.__name__, agg, input_data, expected_data, column)

def run_merge_test(agg: Aggregation, quantized_data: dict[str, Any], expected_data: dict[str, Any], column: str = "val"):
    _run_test(agg.expr_merge.__name__, agg, quantized_data, expected_data, column)

def _run_test(agg_method_name: str, agg: Aggregation, input_data: dict[str, Any], expected_data: dict[str, Any], column: str = "val"):
    def test_func(df: Frame) -> pl.DataFrame | pl.LazyFrame:
        exprs = getattr(agg, agg_method_name)(column)
        if not isinstance(exprs, list):
            exprs = [exprs]

        # Use a dummy group to simulate the behavior in merge_quantized_impl
        return df.group_by(pl.lit(1).alias("_dummy")).agg(exprs).drop("_dummy")

    run_test_twice(test_func, input_data, expected_data)
