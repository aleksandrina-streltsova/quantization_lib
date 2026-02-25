import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quantization._impl.quantization import _calculate_best_chunk_size_impl
from quantization.aggregation import AggCircularMean, AggValueCounts, AggregationPlan
from quantization.api import collect_frame_statistics, finalize_quant_column_configs, quantize
from quantization.config import QuantizationConfig
from quantization.quant_columns import FixedEdgesQuantColumnConfig, FixedStepQuantColumnConfig, \
    UncertaintyQuantColumnConfig, ProbabilitiesQuantColumnConfig
from quantization.types import FrameStatistics

_ALL_AGGS_QUANTIZE = [
    "min", "max", "mean", "off",
    AggCircularMean(low=-2, high=6),
    AggValueCounts(values=[1, 2, 3, 5]),
]
_ALL_AGGS_MERGE = [
    "min", "max", "mean", "off",
    AggCircularMean(low=-2, high=6),
    AggValueCounts(values=[1, 2, 3, 5, -1, -2]),
]
_SCHEMA_INPUT = pl.Schema({
    "v1": pl.Float64,
    "v2": pl.Float64,
    "v3": pl.Int64,
    "v4": pl.Int64,
})


def _build_config(scenario: str, all_aggs: list) -> QuantizationConfig:
    per_column = {}
    default_non_quant = []
    default_quant = None

    if scenario == "single_per_column":
        per_column = {"v1": "min"}
    elif scenario == "single_default_non_quant":
        default_non_quant = "min"
    elif scenario == "single_default_quant":
        default_quant = "max"
    elif scenario == "all":
        per_column = {"v1": "min"}
        default_non_quant = all_aggs
        default_quant = ["max"]

    return QuantizationConfig(
        quant_column_configs=[
            FixedStepQuantColumnConfig("v1", 1.0),
            FixedStepQuantColumnConfig("v2", 1.0),
        ],
        agg_plan=AggregationPlan(
            per_column=per_column,
            default_non_quant=default_non_quant,
            default_quant=default_quant,
        ),
        output_dir=".",
    )


@pytest.mark.parametrize(
    ("scenario", "expected_chunk_size"),
    [
        ("no_aggregations", 51),
        ("single_per_column", 36),
        ("single_default_non_quant", 28),
        ("single_default_quant", 28),
        ("all", 7),
    ],
)
def test_calculate_best_chunk_size_impl_quantize_scenarios(scenario, expected_chunk_size):
    config = _build_config(scenario, _ALL_AGGS_QUANTIZE)

    available_memory_gb = 2 ** -20  # 1 KB -> 2 ** 10 B
    chunk_size = _calculate_best_chunk_size_impl(_SCHEMA_INPUT, config, available_memory_gb, is_merge=False)

    assert chunk_size == expected_chunk_size


@pytest.mark.parametrize(
    ("scenario", "expected_chunk_size"),
    [
        ("no_aggregations", 51),
        ("single_per_column", 36),
        ("single_default_non_quant", 28),
        ("single_default_quant", 28),
        ("all", 4),
    ],
)
def test_calculate_best_chunk_size_impl_merge_scenarios(scenario, expected_chunk_size):
    config = _build_config(scenario, _ALL_AGGS_MERGE)
    frame_empty = pl.DataFrame(schema=_SCHEMA_INPUT)
    frame_quant = quantize(frame_empty, config)
    schema = frame_quant.schema

    available_memory_gb = 2 ** -20  # 1 KB -> 2 ** 10 B
    chunk_size = _calculate_best_chunk_size_impl(schema, config, available_memory_gb, is_merge=True)

    assert chunk_size == expected_chunk_size


@pytest.mark.parametrize("ecdf_columns", [None, ["q1", "aux"]])
def test_collect_frame_statistics_with_optional_ecdf_columns(ecdf_columns):
    df = pl.DataFrame({
        "q1": [0.1, 0.12, 0.2, 0.3, 0.5, 0.48],
        "q2": [0.99, 1.0, 2.0, 2.0, 3.01, 3.02],
        "aux": [10.0, 20.0, 20.0, 30.0, 30.0, 40.0],
    })

    factors = [1.0, 2.0]

    stats = collect_frame_statistics(
        frame=df,
        quant_columns=["q1", "q2"],
        uncertainties={"q1": 0.1, "q2": 0.2},
        ecdf_columns=ecdf_columns,
        factors=factors,
        available_memory_gb=None,
        chunk_size=2,
    )


    expected_ecdf_cols = {"q1", "q2"} if ecdf_columns is None else set(ecdf_columns)
    assert set(stats.column2ecdf.keys()) == expected_ecdf_cols
    assert set(stats.factor2size.keys()) == set(factors)
    assert all(size > 0 for size in stats.factor2size.values())

    expected_ecdfs = {
        "q1": pl.DataFrame({
            "q1": [0.1, 0.12, 0.2, 0.3, 0.48, 0.5],
            "count": [1, 1, 1, 1, 1, 1],
            "quantile": [i / 6 for i in range(1, 7)],
        }).with_columns(pl.col("count").cast(pl.UInt64)),
        "q2": pl.DataFrame({
            "q2": [1.0, 2.0, 3.0, 3.02],
            "count": [2, 2, 1, 1],
            "quantile": [1 / 3, 2 / 3, 5 / 6, 1.0],
        }).with_columns(pl.col("count").cast(pl.UInt64)),
        "aux": pl.DataFrame({
            "aux": [10.0, 20.0, 30.0, 40.0],
            "count": [1, 2, 2, 1],
            "quantile": [1 / 6, 1 / 2, 5 / 6, 1.0],
        }).with_columns(pl.col("count").cast(pl.UInt64)),
    }

    for col in stats.column2ecdf:
        assert_frame_equal(
            stats.column2ecdf[col].sort(col),
            expected_ecdfs[col],
            check_exact=False,
            atol=1e-6,
            rtol=1e-6,
        )

    assert stats.factor2size == {1.0: 4, 2.0: 5}


def test_finalize_quant_column_configs_uncertainty_and_finalized_columns():
    frame_stats = FrameStatistics(
        column2ecdf={},
        factor2size={1.0: 120, 2.0: 80, 4.0: 40},
    )
    configs = [
        UncertaintyQuantColumnConfig(name="u", uncertainty=0.5),
        FixedStepQuantColumnConfig(name="z", step=1.0),
    ]

    finalized = finalize_quant_column_configs(
        frame=pl.DataFrame({"u": [1.0], "z": [2.0]}),
        configs=configs,
        max_n_final=100,
        frame_stats=frame_stats,
    )
    by_name = {cfg.name: cfg for cfg in finalized}

    assert isinstance(by_name["u"], FixedStepQuantColumnConfig)
    assert by_name["u"].step == 1.0
    assert isinstance(by_name["z"], FixedStepQuantColumnConfig)
    assert by_name["z"].step == 1.0


def test_finalize_quant_column_configs_probabilities_and_finalized_columns():
    frame_stats = FrameStatistics(
        column2ecdf={"p": pl.Series("p", [0.0, 10.0, 20.0, 30.0, 40.0])},
        factor2size={1.0: 10},
    )
    configs = [
        ProbabilitiesQuantColumnConfig(name="p", probabilities=np.array([0.0, 0.5, 1.0])),
        FixedStepQuantColumnConfig(name="z", step=2.0),
    ]

    finalized = finalize_quant_column_configs(
        frame=pl.DataFrame({"p": [1.0], "z": [2.0]}),
        configs=configs,
        frame_stats=frame_stats,
    )
    by_name = {cfg.name: cfg for cfg in finalized}

    assert isinstance(by_name["p"], FixedEdgesQuantColumnConfig)
    assert np.all(by_name["p"].edges == [0.0, 20.0, 40.0])
    assert isinstance(by_name["z"], FixedStepQuantColumnConfig)
    assert by_name["z"].step == 2.0


def test_finalize_quant_column_configs_uncertainty_and_probabilities_together():
    frame_stats = FrameStatistics(
        column2ecdf={"p": pl.Series("p", [0.0, 5.0, 10.0, 15.0])},
        factor2size={1.0: 100, 2.0: 60},
    )
    configs = [
        UncertaintyQuantColumnConfig(name="u", uncertainty=0.25),
        ProbabilitiesQuantColumnConfig(name="p", probabilities=np.array([0.0, 1.0])),
    ]

    finalized = finalize_quant_column_configs(
        frame=pl.DataFrame({"u": [0.0], "p": [1.0]}),
        configs=configs,
        max_n_final=80,
        frame_stats=frame_stats,
    )
    by_name = {cfg.name: cfg for cfg in finalized}

    assert isinstance(by_name["u"], FixedStepQuantColumnConfig)
    assert by_name["u"].step == 0.5
    assert isinstance(by_name["p"], FixedEdgesQuantColumnConfig)
    assert np.all(by_name["p"].edges == [0.0, 15.0])
