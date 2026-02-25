import pathlib
import logging
from collections import deque, defaultdict
from typing import Iterable, Any, Iterator

import polars as pl
from quantization._impl.logging_helpers import _QuantizationLogger
from quantization.aggregation import AggregationKey, Aggregation
from quantization.config import QuantizationConfig
from quantization.constants import COLUMN_COUNT, COLUMN_SUFFIX_QUANT
from quantization.types import Frame, QuantizationStep


def quantize_impl(frame: Frame, config: QuantizationConfig) -> Frame:
    q_logger = _QuantizationLogger(QuantizationStep.QUANTIZE)
    frame = q_logger.log_before(frame)

    frame_result = _round(frame, config)
    frame_result = _aggregate(frame_result, config)

    frame_result = q_logger.log_after(frame_result)

    return frame_result


def merge_quantized_impl(frames_quant: Iterable[Frame], config: QuantizationConfig) -> Frame:
    frame = pl.concat(frames_quant, how="diagonal")
    if isinstance(frame, pl.DataFrame) and frame.is_empty():
        return frame

    q_logger = _QuantizationLogger(QuantizationStep.MERGE)
    frame = q_logger.log_before(frame)

    quant_columns = config.quant_columns()
    quant_columns_quant = config.quant_columns(with_suffix=True)
    exprs = _collect_exprs(config, frame, quant_columns, agg_method_name=Aggregation.expr_merge.__name__)
    exprs.append(pl.col(COLUMN_COUNT).sum())
    frame_result = frame.group_by(quant_columns_quant).agg(exprs)

    frame_result = q_logger.log_after(frame_result)

    return frame_result


def quantize_streaming_impl(
    frame: Frame,
    config: QuantizationConfig,
    available_memory_gb: float | None = None,
    chunk_size: int | None = None,
) -> None:
    assert chunk_size is not None or available_memory_gb is not None, \
        "Either `chunk_size` or `available_memory_gb` must be provided."
    if chunk_size is None:
        schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
        chunk_size = _calculate_best_chunk_size_impl(schema, config, available_memory_gb, is_merge=False)
    assert chunk_size > 0, "`chunk_size` must be positive."

    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_iterator = _get_frame_iterator(frame, chunk_size)

    for i, chunk in enumerate(frame_iterator):
        chunk = chunk.collect() if isinstance(chunk, pl.LazyFrame) else chunk
        quantized = quantize_impl(chunk, config)
        quantized.write_parquet(output_dir / f"quantized_{i}.parquet")


def merge_streaming_impl(
    frames: Iterable[Frame],
    config: QuantizationConfig,
    available_memory_gb: float | None = None,
    chunk_size: int | None = None,
) -> pl.DataFrame:
    frames = list(frames)
    if not frames:
        return pl.DataFrame()
    if len(frames) == 1:
        return frames[0] if isinstance(frames[0], pl.DataFrame) else frames[0].collect()

    assert chunk_size is not None or available_memory_gb is not None, \
        "Either `chunk_size` or `available_memory_gb` must be provided."
    if chunk_size is None:
        schema = frames[0].collect_schema() if isinstance(frames[0], pl.LazyFrame) else frames[0].schema
        chunk_size = _calculate_best_chunk_size_impl(schema, config, available_memory_gb, is_merge=True)
    assert chunk_size > 0, "`chunk_size` must be positive."

    level = 0
    frames_curr_level = frames
    while len(frames_curr_level) > 1:
        frames_curr_level = _merge_level(frames_curr_level, config, level, chunk_size)
        level += 1
    final = frames_curr_level[0].collect()

    return final


def _round(frame: Frame, config: QuantizationConfig) -> Frame:
    return frame.with_columns([col_config.expr_round().name.suffix(COLUMN_SUFFIX_QUANT)
                               for col_config in config.quant_column_configs])


def _aggregate(frame: Frame, config: QuantizationConfig) -> Frame:
    quant_columns = config.quant_columns()
    quant_columns_quant = config.quant_columns(with_suffix=True)

    exprs = _collect_exprs(config, frame, quant_columns, agg_method_name=Aggregation.expr_quantize.__name__)
    exprs.append(pl.len().alias(COLUMN_COUNT))

    frame_result = frame.group_by(quant_columns_quant).agg(exprs)

    return frame_result


def _collect_exprs(
    config: QuantizationConfig, frame: Frame,
    quant_columns: list[str],
    agg_method_name: str,
) -> list[Any] | Any:
    def flatten_exprs(agg_keys: AggregationKey | list[AggregationKey], column: str) -> list[pl.Expr]:
        agg_keys = _to_iterable(agg_keys)
        aggs = [config.agg_registry[agg_key] if isinstance(agg_key, str) else agg_key for agg_key in agg_keys]

        return _flatten([getattr(agg, agg_method_name)(column) for agg in aggs])

    columns = frame.collect_schema().names() if isinstance(frame, pl.LazyFrame) else frame.columns
    columns = set(filter(lambda x: x is not None, [Aggregation.get_base_name(col) for col in columns]))

    exprs = []
    for col in columns:
        # 1. Add aggregations for columns specified in `per_column`.
        if col in config.agg_plan.per_column:
            exprs += flatten_exprs(config.agg_plan.per_column[col], col)

        # 2. Add default aggregations for quantization columns not specified in `per_column`.
        elif col in quant_columns:
            if config.agg_plan.default_quant is not None:
                exprs += flatten_exprs(config.agg_plan.default_quant, col)

        # 3. Add default aggregations for non-quantization columns not specified in `per_column`.
        else:
            exprs += flatten_exprs(config.agg_plan.default_non_quant, col)

    return exprs


def _calculate_best_chunk_size_impl(
    schema: pl.Schema,
    config: QuantizationConfig,
    available_memory_gb: float,
    is_merge: bool,
) -> int:
    assert available_memory_gb > 0, "Available memory must be positive."
    schema_dict = dict(schema.items())

    # 1. Collect memory requirements for quantization columns.
    quant_columns = config.quant_columns(with_suffix=is_merge)
    bytes_quant = sum(_dtype_size_bytes(schema.get(col)) for col in quant_columns if col in schema_dict)

    # 2. Collect memory requirements for aggregations.
    base2col_names = defaultdict(dict)
    for col_name, dtype in schema.items():
        if col_name in quant_columns:
            continue

        base_name = Aggregation.get_base_name(col_name)
        if base_name is None:
            continue

        base2col_names[base_name][col_name] = dtype

    quant_columns_base = config.quant_columns()
    bytes_agg = 0.0
    bytes_agg += _dtype_size_bytes(pl.UInt32)  # COUNT_COLUMN
    for base_name, col_name2dtype in base2col_names.items():
        if base_name in config.agg_plan.per_column:
            agg_keys = config.agg_plan.per_column[base_name]
        elif base_name in quant_columns_base:
            if config.agg_plan.default_quant is None:
                continue
            agg_keys = config.agg_plan.default_quant
        else:
            agg_keys = config.agg_plan.default_non_quant

        agg_keys_iter = _to_iterable(agg_keys)
        aggs = [config.agg_registry[agg_key] if isinstance(agg_key, str) else agg_key for agg_key in agg_keys_iter]
        for agg in aggs:
            bytes = sum(_dtype_size_bytes(dtype)
                        for col_name, dtype in col_name2dtype.items()
                        if col_name.startswith(f"{base_name}{agg.suffix}"))
            bytes_agg += bytes * agg.memory_factor(is_merge)


    bytes_total = bytes_quant + bytes_agg
    assert bytes_total > 0, "Total memory requirements for a single row must be positive."

    available_memory_b = available_memory_gb * 2 ** 30
    chunk_size = max(2, int(available_memory_b / bytes_total))
    logging.info("Calculated chunk size: %d (is_merge=%s, available_memory_gb=%.6f, bytes_per_row=%.2f).",
                 chunk_size, is_merge, available_memory_gb, bytes_total)

    return chunk_size


def _dtype_size_bytes(dtype: Any) -> int:
    base_dtype = dtype.base_type() if hasattr(dtype, "base_type") else dtype

    match base_dtype:
        case pl.Int8 | pl.UInt8 | pl.Boolean:
            return 1
        case pl.Int16 | pl.UInt16:
            return 2
        case pl.Int32 | pl.UInt32 | pl.Float32 | pl.Date | pl.Categorical | pl.Enum:
            return 4
        case pl.Int64 | pl.UInt64 | pl.Float64 | pl.Datetime | pl.Duration | pl.Time:
            return 8
        case pl.Decimal:
            return 16
        case pl.String | pl.Binary | pl.List | pl.Array | pl.Struct | pl.Object | pl.Unknown:
            return 32
        case pl.Null:
            return 1

    return 16


def _merge_level(frames: list[Frame], config: QuantizationConfig, level: int, chunk_size: int) -> list[pl.LazyFrame]:
    output_dir = pathlib.Path(config.output_dir)

    n_prev_level = len(frames)
    n_curr_level = 0

    frames = deque(frames)
    frames_merged = []

    frames_to_merge = []
    current_size = 0

    while frames:
        frame = frames.popleft()
        frame = frame.collect() if isinstance(frame, pl.LazyFrame) else frame

        frames_to_merge.append(frame)
        current_size += frame.height

        if frames and (len(frames_to_merge) <= 1 or current_size + frame.height <= chunk_size):
            continue

        merged = merge_quantized_impl(frames_to_merge, config)

        if len(frames_to_merge) == n_prev_level:
            filepath = output_dir / f"merged_final.parquet"

        else:
            level_dir = output_dir / f"level_{level}"
            level_dir.mkdir(parents=True, exist_ok=True)
            filepath = level_dir / f"merged_{n_curr_level}.parquet"

        merged.write_parquet(filepath)
        frames_merged.append(pl.scan_parquet(filepath))

        n_curr_level += 1
        frames_to_merge = []
        current_size = 0

    return frames_merged


def _to_iterable(x) -> Iterable:
    if not isinstance(x, str) and isinstance(x, Iterable):
        return x
    return [x]


def _flatten(x: Iterable) -> list:
    result = []

    for item in x:
        if isinstance(item, Iterable):
            result += _flatten(item)
        else:
            result.append(item)

    return result


def _get_frame_iterator(frame: Frame, chunk_size: int) -> Iterator[Frame]:
    if isinstance(frame, pl.LazyFrame):
        frame_iterator = frame.collect_batches(chunk_size=chunk_size, maintain_order=True)
    else:
        frame_iterator = iter([frame.slice(offset, chunk_size) for offset in range(0, frame.height, chunk_size)])

    return frame_iterator
