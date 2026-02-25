from typing import Any

from quantization.types import Frame
from quantization._impl.quantization import _round
from quantization.config import QuantizationConfig
from quantization.quant_columns import FixedStepQuantColumnConfig, FixedEdgesQuantColumnConfig, \
    FinalizedQuantColumnConfig, RoundingQuantColumnConfig
from tests.utils import run_test_twice


def _run_test(col_config: FinalizedQuantColumnConfig, input_data: dict[str, Any], expected_data: dict[str, Any]):
    def test_func(df: Frame) -> Frame:
        config = QuantizationConfig(
            quant_column_configs=[col_config],
            agg_plan=None,
            output_dir=".",
        )
        return _round(df, config)

    run_test_twice(test_func, input_data, expected_data)


def test_fixed_step_round():
    config = FixedStepQuantColumnConfig(name="val", step=2.0)
    input_data = {
        "val": [1.1, 2.1, 2.9, 4.1, 5.9, None],
    }
    expected_data = {
        "val": [1.1, 2.1, 2.9, 4.1, 5.9, None],
        "val__quant": [2.0, 2.0, 2.0, 4.0, 6.0, None],
    }

    _run_test(config, input_data, expected_data)


def test_fixed_step_round_use_integer_indices():
    config = FixedStepQuantColumnConfig(name="val", step=2.0, use_integer_indices=True)
    input_data = {
        "val": [1.1, 2.1, 2.9, 4.1, 5.9, None],
    }
    expected_data = {
        "val": [1.1, 2.1, 2.9, 4.1, 5.9, None],
        "val__quant": [1, 1, 1, 2, 3, None],
    }

    _run_test(config, input_data, expected_data)


def test_decimals_round():
    config = RoundingQuantColumnConfig(name="val", decimals=1)
    input_data = {
        "val": [0.2, 1.213, 3.31, 73.3, 110.3, 5.5, 6.09, None],
    }
    expected_data = {
        "val": [0.2, 1.213, 3.31, 73.3, 110.3, 5.5, 6.09, None],
        "val__quant": [0.2, 1.2, 3.3, 73.3, 110.3, 5.5, 6.1, None],
    }

    _run_test(config, input_data, expected_data)


def test_fixed_edges_round():
    config = FixedEdgesQuantColumnConfig(name="val", edges=[0, 2, 5, 10])
    input_data = {
        "val": [-1, 1, 1.5, 3, 4, 6, 8, 12, None],
    }
    expected_data = {
        "val": [-1, 1, 1.5, 3, 4, 6, 8, 12, None],
        "val__quant": [0, 1, 1, 3.5, 3.5, 7.5, 7.5, 10, None],
    }

    _run_test(config, input_data, expected_data)


def test_fixed_edges_round_with_clip():
    config = FixedEdgesQuantColumnConfig(name="val", edges=[0, 5, 10], clip=(2, 6))
    input_data = {
        "val": [0, 1, 3, 7, 11, 5.5, None],
    }
    expected_data = {
        "val": [0, 1, 3, 7, 11, 5.5, None],
        "val__quant": [2, 2, 3.5, 6, 6, 5.5, None],
    }

    _run_test(config, input_data, expected_data)


def test_decimals_round_with_clip():
    config = RoundingQuantColumnConfig(name="val", decimals=0, clip=(0, 6))
    input_data = {
        "val": [0.2, 1.213, 3.31, 73.3, 110.3, 5.5, 6.09, None],
    }
    expected_data = {
        "val": [0.2, 1.213, 3.31, 73.3, 110.3, 5.5, 6.09, None],
        "val__quant": [0., 1., 3., 6., 6., 6., 6., None],
    }

    _run_test(config, input_data, expected_data)
