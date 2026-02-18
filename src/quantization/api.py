from typing import Iterable

from quantization._impl.quantization import (
    quantize_impl, merge_quantized_impl,
    quantize_streaming_impl,
    merge_streaming_impl
)
from quantization._impl.finalizing import finalize_quant_column_configs_impl, collect_frame_statistics_impl
from quantization.config import QuantizationConfig
from quantization.quant_columns import _BaseQuantColumnConfig, FinalizedQuantColumnConfig
from quantization.types import Frame,  FrameStatistics


def finalize_quant_column_configs(frame: Frame,
                                  configs: Iterable[_BaseQuantColumnConfig],
                                  max_n_final: int | None = None,
                                  frame_stats: FrameStatistics | None = None) -> list[FinalizedQuantColumnConfig]:
    """
    Configurations may be unfinalized if they need data-dependent computations.
    If there are unfinalized configurations, an additional data pass will be performed to finalize them.

    Args:
        frame: The input data serving as the basis for finalizing configurations.
        configs: Column configurations, some of which may require finalization using the provided data.
        max_n_final: Maximum number of final signatures allowed for uncertainty-based columns.
        frame_stats: Optional pre-computed frame statistics.

    Returns:
        list[FinalizedQuantColumnConfig]: Finalized column configurations.
    """
    return finalize_quant_column_configs_impl(frame, configs, max_n_final, frame_stats)


def collect_frame_statistics(
    frame: Frame,
    quant_columns: list[str],
    uncertainties: dict[str, float],
    ecdf_columns: list[str] | None = None,
    factors: list[float] | None = None,
    available_memory_gb: float | None = None,
    chunk_size: int | None = None,
) -> FrameStatistics:
    """
    Collects statistics used for quantization-config finalization.

    Computes:
        1) empirical CDF approximations per column;
        2) estimated number of unique quantized signatures for each tested factor.

    Args:
        frame: Input frame to scan.
        quant_columns: Column names used as quantization dimensions.
        uncertainties: Per-column uncertainty values used to derive factor-specific steps.
        ecdf_columns: Columns for ECDF calculation. If None, `quant_columns` are used.
        factors: Scaling factors to evaluate. If None, defaults are used.
        available_memory_gb: Available memory budget in GB for auto chunk-size estimation.
        chunk_size: Explicit chunk size (number of rows). If provided, takes precedence over memory-based estimation.

    Returns:
        FrameStatistics: Collected ECDF and factor-to-size estimates.
    """
    return collect_frame_statistics_impl(
        frame=frame,
        quant_columns=quant_columns,
        uncertainties=uncertainties,
        ecdf_columns=ecdf_columns,
        factors=factors,
        available_memory_gb=available_memory_gb,
        chunk_size=chunk_size,
    )


def quantize(frame: Frame, config: QuantizationConfig) -> Frame:
    """
    Quantizes the data using provided config.

    Steps (conceptual):
        1) Clip quantization columns if ranges are provided.
        2) Apply rounding/binning:
            - fixed_step: round to nearest step.
            - fixed_edges: bucketize by edges.
        3) Group by the resulting quantized columns ("signature").
        4) Apply per-column aggregations based on :py:func:~quantization.aggregation.Aggregation.expr_quantize().

    Args:
        frame: The input data to quantize.
        config: Config with finalized column configurations.

    Returns:
        Frame: Quantized/distilled frame consisting of signature columns and aggregated columns.
    """
    return quantize_impl(frame, config)


def merge_quantized(frames_quant: Iterable[Frame], config: QuantizationConfig) -> Frame:
    """
    Merges previously quantized frames using provided config.

    Steps (conceptual):
        1) Group by the resulting quantized columns ("signature").
        2) Apply per-column aggregations based on :py:func:~quantization.aggregation.Aggregation.expr_merge().

    Args:
        frames_quant: Pre-quantized frames with the same signature columns.
        config: Config with finalized column configurations.

    Returns:
        Frame: Merged quantized/distilled frame consisting of signature columns and aggregated columns.
    """
    return merge_quantized_impl(frames_quant, config)


def quantize_streaming(
    frame: Frame,
    config: QuantizationConfig,
    available_memory_gb: float | None = None,
    chunk_size: int | None = None,
):
    """
    Performs streaming quantization of the data.

    Either `chunk_size` or `available_memory_gb` must be provided.
    If both are provided, `chunk_size` is used.
    """
    return quantize_streaming_impl(frame, config, available_memory_gb, chunk_size)


def merge_streaming(
    frames_quant: Iterable[Frame],
    config: QuantizationConfig,
    available_memory_gb: float | None = None,
    chunk_size: int | None = None,
) -> Frame:
    """
    Performs streaming merge of the data.

    Either `chunk_size` or `available_memory_gb` must be provided.
    If both are provided, `chunk_size` is used.
    """
    return merge_streaming_impl(frames_quant, config, available_memory_gb, chunk_size)


def select_initial_rows_by_quantized(frame_init: Frame, frame_quant: Frame, config: QuantizationConfig) -> Frame:
    """
    Selects original rows from frame_init that correspond to signatures present in frame_quant.

    Steps (conceptual):
        1) Recompute (or reuse) the same rounding/binning for the signature calculation on frame_init.
        2) Join against frame_quant's signature keys.
        3) Return the subset of frame_init rows whose signatures match.

    Args:
        frame_init: Original, un-quantized data.
        frame_quant: Quantized frame (its signature defines which original rows to select).
        config: Fully resolved config.

    Returns:
        Frame: initial rows matching the input signatures.
    """
    raise NotImplementedError
