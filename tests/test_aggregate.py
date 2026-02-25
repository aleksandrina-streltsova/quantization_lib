from typing import Any

from quantization._impl.quantization import _aggregate
from quantization.aggregation import AggValueCounts, AggCircularMean, AggregationPlan
from quantization.config import QuantizationConfig
from quantization.constants import COLUMN_COUNT
from quantization.quant_columns import FinalizedQuantColumnConfig, FixedStepQuantColumnConfig
from quantization.types import Frame
from tests.utils import run_test_twice


def _run_test(
    input_data: dict[str, Any], expected_data: dict[str, Any],
    quant_column_configs: list[FinalizedQuantColumnConfig],
    agg_per_column=None, agg_default_non_quant="mean", agg_default_quant=None,
):
    def test_func(df: Frame) -> Frame:
        config = QuantizationConfig(
            quant_column_configs,
            agg_plan=AggregationPlan(
                per_column=agg_per_column or {},
                default_non_quant=agg_default_non_quant,
                default_quant=agg_default_quant,
            ),
            output_dir=".",
        )
        return _aggregate(df, config)

    run_test_twice(test_func, input_data, expected_data)


def test_single_quant_default_agg():
    quant_column_configs = [FixedStepQuantColumnConfig("val", 1)]
    input_data = {
        "val": [1.1, 1.2, 2.1, None],
        "val__quant": [1., 1., 2., None],
        "other": [10, 20, 30, 40]
    }

    agg_default_non_quant = "min"

    expected_data = {
        "val__quant": [None, 1, 2],
        "other__min": [40, 10, 30],
        COLUMN_COUNT: [1, 2, 1]
    }

    return _run_test(input_data, expected_data, quant_column_configs, agg_default_non_quant=agg_default_non_quant)


def test_multiple_quant_default_agg():
    quant_column_configs = [
        FixedStepQuantColumnConfig("v1", 1),
        FixedStepQuantColumnConfig("v2", 1)
    ]
    input_data = {
        "v1": [1.1, 1.2, 2.1, 2.1, None],
        "v1__quant": [1., 1., 2., 2., None],
        "v2": [10.1, 10.2, 20.1, 10.1, 30.1],
        "v2__quant": [10., 10., 20., 10., 30.],
        "v3": [1, 2, 3, -1, 5],
        "v4": [1, 2, 2, 2, -2],
    }

    agg_default_non_quant = [
        "min", "max", "mean", "off",
        AggCircularMean(low=-2, high=6),
        AggValueCounts(values=[1, 2, 3, 5]),
    ]

    expected_data = {
        "v1__quant": [None, 1, 2, 2],
        "v2__quant": [30, 10, 10, 20],
        "v3__min": [5, 1, -1, 3],
        "v4__min": [-2, 1, 2, 2],
        "v3__max": [5, 2, -1, 3],
        "v4__max": [-2, 2, 2, 2],
        "v3__mean": [5, 1.5, -1, 3],
        "v3__meancount": [1, 2, 1, 1],
        "v4__mean": [-2, 1.5, 2, 2],
        "v4__meancount": [1, 2, 1, 1],
        "v3__circmean": [5, 1.5, -1, 3],
        "v3__circmeancount": [1, 2, 1, 1],
        "v4__circmean": [-2, 1.5, 2, 2],
        "v4__circmeancount": [1, 2, 1, 1],
        "v3__off": [[5], [1, 2], [-1], [3]],
        "v4__off": [[-2], [1, 2], [2], [2]],
        "v3__valuecounts": [
            [{"v3": 5, "struct_count": 1}],
            [{"v3": 1, "struct_count": 1}, {"v3": 2, "struct_count": 1}],
            [{"v3": -1, "struct_count": 1}],
            [{"v3": 3, "struct_count": 1}],
        ],
        "v4__valuecounts": [
            [{"v4": -2, "struct_count": 1}],
            [{"v4": 1, "struct_count": 1}, {"v4": 2, "struct_count": 1}],
            [{"v4": 2, "struct_count": 1}],
            [{"v4": 2, "struct_count": 1}],
        ],
        COLUMN_COUNT: [1, 2, 1, 1]
    }

    return _run_test(input_data, expected_data, quant_column_configs, agg_default_non_quant=agg_default_non_quant)


def test_single_quant_default_quant_agg():
    quant_column_configs = [FixedStepQuantColumnConfig("val", 1)]
    input_data = {
        "val": [1.1, 1.2, 2.1, None],
        "val__quant": [1., 1., 2., None],
        "other": [10, 20, 30, 40]
    }

    agg_default_non_quant = "min"
    agg_default_quant = "max"

    expected_data = {
        "val__quant": [None, 1, 2],
        "val__max": [None, 1.2, 2.1],
        "other__min": [40, 10, 30],
        COLUMN_COUNT: [1, 2, 1]
    }

    return _run_test(input_data, expected_data, quant_column_configs,
                     agg_default_non_quant=agg_default_non_quant,
                     agg_default_quant=agg_default_quant)


def test_per_column_single_agg():
    quant_column_configs = [FixedStepQuantColumnConfig("v1", 1)]
    input_data = {
        "v1": [1.1, 1.1, None],
        "v1__quant": [1., 1., None],
        "v2": [100, 200, 300]
    }

    agg_per_column = {
        "v2": "min"
    }
    agg_default_non_quant = "min"

    expected_data = {
        "v1__quant": [None, 1],
        "v2__min": [300, 100],
        COLUMN_COUNT: [1, 2]
    }

    return _run_test(
        input_data, expected_data, quant_column_configs,
        agg_per_column=agg_per_column,
        agg_default_non_quant=agg_default_non_quant,
    )


def test_per_column_multiple_agg():
    quant_column_configs = [FixedStepQuantColumnConfig("v1", 1)]
    input_data = {
        "v1": [1.1, 1.1, None],
        "v1__quant": [1., 1., None],
        "v2": [10, 20, 30],
        "v3": [100, 200, 300]
    }

    agg_per_column = {
        "v2": ["min", "max"],
        "v3": "min"
    }
    agg_default_non_quant = "min"

    expected_data = {
        "v1__quant": [None, 1],
        "v2__min": [30, 10],
        "v2__max": [30, 20],
        "v3__min": [300, 100],
        COLUMN_COUNT: [1, 2]
    }

    return _run_test(
        input_data, expected_data, quant_column_configs,
        agg_per_column=agg_per_column,
        agg_default_non_quant=agg_default_non_quant,
    )


def test_multiple_quant_per_column_multiple_agg_default_quant_agg():
    quant_column_configs = [
        FixedStepQuantColumnConfig("v1", 1),
        FixedStepQuantColumnConfig("v2", 1)
    ]
    input_data = {
        "v1": [1.1, 1.1, 1.1, 2.1, 2.3, None],
        "v1__quant": [1., 1., 1., 2., 2., None],
        "v2": [10.1, 10.1, 10.1, 10.1, 20.1, 30.1],
        "v2__quant": [10., 10., 10., 10., 20., 30.],
        "v3": [100, 200, None, 300, 200, 400],
        "v4": ["a", "b", "b", "a", "c", None]
    }

    agg_per_column = {"v1": "max",
                      "v3": ["min", "mean"], }
    agg_default_quant = ["off"]
    agg_default_non_quant = [AggValueCounts(values=["a", "b", "c", None])]

    expected_data = {
        "v1__quant": [None, 1, 2, 2],
        "v2__quant": [30, 10, 10, 20],
        "v1__max": [None, 1.1, 2.1, 2.3],
        "v2__off": [[30.1], [10.1, 10.1, 10.1], [10.1], [20.1]],
        "v3__min": [400.0, 100.0, 300.0, 200.0],
        "v3__mean": [400.0, 150.0, 300.0, 200.0],
        "v3__meancount": [1, 2, 1, 1],
        "v4__valuecounts": [
            [
                {"v4": None, "struct_count": 1},
            ],
            [
                {"v4": "a", "struct_count": 1},
                {"v4": "b", "struct_count": 2},
            ],
            [
                {"v4": "a", "struct_count": 1},
            ],
            [
                {"v4": "c", "struct_count": 1},
            ]
        ],
        COLUMN_COUNT: [1, 3, 1, 1]
    }

    return _run_test(
        input_data, expected_data, quant_column_configs,
        agg_per_column=agg_per_column,
        agg_default_quant=agg_default_quant,
        agg_default_non_quant=agg_default_non_quant
    )
