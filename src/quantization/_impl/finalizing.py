from collections import defaultdict
import logging
from typing import Iterable

import hll_rust_py
import polars as pl
from quantization._impl.quantization import _get_frame_iterator, quantize_impl, \
    _calculate_best_chunk_size_impl
from quantization._impl.rounding import expr_round_fixed_step_impl
from quantization.aggregation import AggregationPlan
from quantization.config import QuantizationConfig
from quantization.constants import COLUMN_SUFFIX_QUANT
from quantization.quant_columns import UncertaintyQuantColumnConfig, FixedStepQuantColumnConfig, _BaseQuantColumnConfig, \
    FinalizedQuantColumnConfig, \
    UnfinalizedQuantColumnConfig
from quantization.types import FrameStatistics, Frame

_DEFAULT_CHUNK_SIZE = 100_000


def finalize_quant_column_configs_impl(
    frame: Frame,
    configs: Iterable[_BaseQuantColumnConfig],
    max_n_final: int | None = None,
    frame_stats: FrameStatistics | None = None
) -> list[FinalizedQuantColumnConfig]:
    unfinalized, finalized = [], []
    for config in configs:
        (unfinalized if isinstance(config, UnfinalizedQuantColumnConfig) else finalized).append(config)

    if not unfinalized:
        return finalized

    if frame_stats is None:
        uncertainties = {c.name: c.uncertainty for c in unfinalized if isinstance(c, UncertaintyQuantColumnConfig)}
        quant_columns = [c.name for c in unfinalized]
        frame_stats = collect_frame_statistics_impl(frame, quant_columns, uncertainties)

    for config in unfinalized:
        finalized.append(config.finalize(frame_stats, max_n_final=max_n_final))

    return finalized


def collect_frame_statistics_impl(
    frame: Frame,
    quant_columns: list[str],
    uncertainties: dict[str, float],
    ecdf_columns: list[str] | None = None,
    factors: list[float] | None = None,
    available_memory_gb: float | None = None,
    chunk_size: int | None = None,
) -> FrameStatistics:
    ecdf_columns = quant_columns if ecdf_columns is None else ecdf_columns
    if factors is None:
        factors = [1., 2., 4., 6., 8., 10.]

    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema

    factor2config = {
        f: QuantizationConfig(
            quant_column_configs=[
                FixedStepQuantColumnConfig(quant_column, f * uncertainties[quant_column], use_integer_indices=True)
                for quant_column in quant_columns
            ],
            agg_plan=AggregationPlan(per_column={}, default_non_quant="mean"),
            output_dir=".",
        )
        for f in factors
    }

    column2hists = defaultdict(list)
    factor2hll = defaultdict(lambda: hll_rust_py.PyHLL())

    # We use a minimal config to estimate chunk size for the most expensive factor (min factor = min size reduction).
    if chunk_size is None:
        if available_memory_gb is None:
            chunk_size = _DEFAULT_CHUNK_SIZE
            logging.warning("Neither `chunk_size` nor `available_memory_gb` was provided; "
                            "using default chunk_size=%d.", chunk_size)
        else:
            chunk_size = _calculate_best_chunk_size_impl(schema, factor2config[min(factors)], available_memory_gb,
                                                         is_merge=False)
    logging.info("Using chunk size=%d.", chunk_size)
    frame_iterator = _get_frame_iterator(frame, chunk_size)
    logging.info("Collected frame iterator.")

    for i, chunk in enumerate(frame_iterator):
        chunk = chunk.collect() if isinstance(chunk, pl.LazyFrame) else chunk
        logging.info("Collecting frame statistics for chunk %d: initial_size=%d.", i, chunk.height)

        # 1. Collect hists for ECDF (we'll merge the hists and get ECDF at the end)
        for col in ecdf_columns:
            if col in uncertainties:
                step = uncertainties[col] / 10
            else:
                logging.warning("Uncertainty for column `%s` was not provided; "
                                "using default ECDF histogram step=0.01.", col)
                step = 0.01
            expr = expr_round_fixed_step_impl(col, step, clip=None, use_integer_indices=False)
            hist = chunk.select(expr.alias(col)).to_series().value_counts()
            logging.info(
                "Collected ECDF value_counts for chunk %d, column `%s`: size=%d.",
                i, col, hist.height,
            )
            column2hists[col].append(hist)

        # 2. Update unique signature counts for each factor
        quant_columns_with_suffix = [f"{q}{COLUMN_SUFFIX_QUANT}" for q in quant_columns]
        for f in factors:
            config = factor2config[f]
            quantized = quantize_impl(chunk, config)

            hashes = quantized.select(pl.struct(quant_columns_with_suffix).hash()).to_series()
            factor2hll[f].extend(hashes.to_numpy())

    column2ecdf = {
        col: _calculate_ecdf_from_hists(hists, col)
        for col, hists in column2hists.items()
    }
    factor2size = {f: hll.count() for f, hll in factor2hll.items()}

    return FrameStatistics(column2ecdf=column2ecdf, factor2size=factor2size)


def _calculate_ecdf_from_hists(hists: list[pl.DataFrame], column: str):
    count = pl.col("count")
    hist = pl.concat(hists).group_by(pl.col(column)).agg(count.sum())
    hist = hist.sort(by=column).with_columns(count.cast(pl.UInt64))
    ecdf = hist.with_columns((count.cum_sum() / count.sum()).alias("quantile"))

    return ecdf
