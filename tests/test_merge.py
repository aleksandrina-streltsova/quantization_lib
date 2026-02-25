from typing import Any

import polars as pl

from quantization.aggregation import AggValueCounts, AggCircularMean, AggregationPlan
from quantization.api import merge_quantized
from quantization.config import QuantizationConfig
from quantization.constants import COLUMN_COUNT
from quantization.quant_columns import FinalizedQuantColumnConfig, FixedStepQuantColumnConfig
from quantization.types import Frame
from tests.utils import run_test_twice


def _run_test(
    frames_data: list[dict[str, Any]], expected_data: dict[str, Any],
    quant_column_configs: list[FinalizedQuantColumnConfig],
    agg_per_column=None, agg_default_non_quant="mean", agg_default_quant=None,
):
    def test_func(frames: list[Frame]) -> Frame:
        config = QuantizationConfig(
            quant_column_configs,
            agg_plan=AggregationPlan(
                per_column=agg_per_column or {},
                default_non_quant=agg_default_non_quant,
                default_quant=agg_default_quant,
            ),
            output_dir=".",
        )
        return merge_quantized(frames, config)

    run_test_twice(test_func, frames_data, expected_data)


def test_merge_single_empty_frame():
    quant_column_configs = [FixedStepQuantColumnConfig("val", 1)]
    frames_data = [{
        "val__quant": pl.Series([], dtype=pl.Float64),
        "other__min": pl.Series([], dtype=pl.Int64),
        COLUMN_COUNT: pl.Series([], dtype=pl.UInt32)
    }]

    expected_data = frames_data[0]

    _run_test(frames_data, expected_data, quant_column_configs, agg_default_non_quant="min")


def test_merge_multiple_empty_frames():
    quant_column_configs = [FixedStepQuantColumnConfig("val", 1)]
    empty_frame = {
        "val__quant": pl.Series([], dtype=pl.Float64),
        "other__min": pl.Series([], dtype=pl.Int64),
        COLUMN_COUNT: pl.Series([], dtype=pl.UInt32)
    }
    frames_data = [empty_frame, empty_frame]

    expected_data = empty_frame

    _run_test(frames_data, expected_data, quant_column_configs, agg_default_non_quant="min")


def test_merge_multiple_non_empty_frames():
    quant_column_configs = [
        FixedStepQuantColumnConfig("v1", 1),
        FixedStepQuantColumnConfig("v2", 1)
    ]

    frame1 = {
        "v1__quant": [1.0, 2.0],
        "v1__min": [0.95, 2.1],
        "v2__quant": [10.0, 20.0],
        "v2__max": [10.1, 20.1],
        "v3__min": [1, 3],
        "v3__max": [2, 3],
        "v3__mean": [1.5, 3.0],
        "v3__meancount": [2, 1],
        "v3__circmean": [3.0, 3.0],
        "v3__circmeancount": [2, 1],
        "v3__off": [[1, 2], [3]],
        "v3__valuecounts": [
            [{"v3": 1, "struct_count": 1}, {"v3": 2, "struct_count": 1}],
            [{"v3": 3, "struct_count": 1}]
        ],
        "v4__min": [1, 2],
        "v4__max": [2, 2],
        "v4__mean": [1.5, 2.0],
        "v4__meancount": [2, 1],
        "v4__circmean": [-2.0, 2.0],
        "v4__circmeancount": [2, 1],
        "v4__off": [[1, 2], [2]],
        "v4__valuecounts": [
            [{"v4": 1, "struct_count": 1}, {"v4": 2, "struct_count": 1}],
            [{"v4": 2, "struct_count": 1}]
        ],
        COLUMN_COUNT: [2, 1]
    }

    frame2 = {
        "v1__quant": [1.0, None],
        "v1__min": [0.9, None],
        "v2__quant": [10.0, 30.0],
        "v2__max": [9.9, 30.2],
        "v3__min": [-1, 5],
        "v3__max": [-1, 5],
        "v3__mean": [-1.0, 5.0],
        "v3__meancount": [1, 1],
        "v3__circmean": [-1.0, 5.0],
        "v3__circmeancount": [1, 1],
        "v3__off": [[-1], [5]],
        "v3__valuecounts": [
            [{"v3": -1, "struct_count": 1}],
            [{"v3": 5, "struct_count": 1}]
        ],
        "v4__min": [2, -2],
        "v4__max": [2, -2],
        "v4__mean": [2.0, -2.0],
        "v4__meancount": [1, 1],
        "v4__circmean": [2.0, -2.0],
        "v4__circmeancount": [1, 1],
        "v4__off": [[2], [-2]],
        "v4__valuecounts": [
            [{"v4": 2, "struct_count": 1}],
            [{"v4": -2, "struct_count": 1}]
        ],
        COLUMN_COUNT: [1, 1]
    }

    frames_data = [frame1, frame2]

    agg_per_column = { "v1": ["min"]}
    agg_default_non_quant = [
        "min", "max", "mean", "off",
        AggCircularMean(low=-2, high=6),
        AggValueCounts(values=[1, 2, 3, 5, -1, -2]),
    ]
    agg_default_quant = ["max"]

    expected_data = {
        "v1__quant": [None, 1.0, 2.0],
        "v1__min": [None, 0.9, 2.1],
        "v2__quant": [30.0, 10.0, 20.0],
        "v2__max": [30.2, 10.1, 20.1],
        "v3__min": [5, -1, 3],
        "v4__min": [-2, 1, 2],
        "v3__max": [5, 2, 3],
        "v4__max": [-2, 2, 2],
        "v3__mean": [5.0, (1.5 * 2 + -1.0 * 1) / 3, 3.0],
        "v3__meancount": [1, 3, 1],
        "v4__mean": [-2.0, (1.5 * 2 + 2.0 * 1) / 3, 2.0],
        "v4__meancount": [1, 3, 1],
        "v3__circmean": [5.0, 3.0, 3.0],
        "v3__circmeancount": [1, 3, 1],
        "v4__circmean": [-2.0, -2.0, 2.0],
        "v4__circmeancount": [1, 3, 1],
        "v3__off": [[5], [1, 2, -1], [3]],
        "v4__off": [[-2], [1, 2, 2], [2]],
        "v3__valuecounts": [
            [{"v3": 1, "struct_count": 0}, {"v3": 2, "struct_count": 0}, {"v3": 3, "struct_count": 0},
             {"v3": 5, "struct_count": 1}, {"v3": -1, "struct_count": 0}, {"v3": -2, "struct_count": 0}],
            [{"v3": 1, "struct_count": 1}, {"v3": 2, "struct_count": 1}, {"v3": 3, "struct_count": 0},
             {"v3": 5, "struct_count": 0}, {"v3": -1, "struct_count": 1}, {"v3": -2, "struct_count": 0}],
            [{"v3": 1, "struct_count": 0}, {"v3": 2, "struct_count": 0}, {"v3": 3, "struct_count": 1},
             {"v3": 5, "struct_count": 0}, {"v3": -1, "struct_count": 0}, {"v3": -2, "struct_count": 0}]
        ],
        "v4__valuecounts": [
            [{"v4": 1, "struct_count": 0}, {"v4": 2, "struct_count": 0}, {"v4": 3, "struct_count": 0},
             {"v4": 5, "struct_count": 0}, {"v4": -1, "struct_count": 0}, {"v4": -2, "struct_count": 1}],
            [{"v4": 1, "struct_count": 1}, {"v4": 2, "struct_count": 2}, {"v4": 3, "struct_count": 0},
             {"v4": 5, "struct_count": 0}, {"v4": -1, "struct_count": 0}, {"v4": -2, "struct_count": 0}],
            [{"v4": 1, "struct_count": 0}, {"v4": 2, "struct_count": 1}, {"v4": 3, "struct_count": 0},
             {"v4": 5, "struct_count": 0}, {"v4": -1, "struct_count": 0}, {"v4": -2, "struct_count": 0}]
        ],
        COLUMN_COUNT: [1, 3, 1]
    }

    return _run_test(frames_data, expected_data, quant_column_configs,
                     agg_per_column, agg_default_non_quant, agg_default_quant)
