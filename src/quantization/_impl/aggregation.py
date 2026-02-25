import math
from typing import Iterable

import polars as pl

from quantization.constants import STRUCT_FIELD_COUNT


def expr_quantize_simple_impl(column: str, op: str, suffix: str) -> pl.Expr:
    return getattr(pl.col(column), op)().alias(f"{column}{suffix}")


def expr_merge_simple_impl(column: str, op: str, suffix: str) -> pl.Expr:
    return getattr(pl.col(f"{column}{suffix}"), op)()


def expr_quantize_off_impl(column: str, suffix: str) -> pl.Expr:
    return pl.col(column).alias(f"{column}{suffix}")


def expr_merge_off_impl(column: str, suffix: str) -> pl.Expr:
    return pl.col(f"{column}{suffix}").list.explode()


def expr_quantize_mean_impl(column: str, suffix: str) -> list[pl.Expr]:
    return [
        pl.col(column).mean().alias(f"{column}{suffix}"),
        (pl.len() - pl.col(column).null_count()).alias(f"{column}{suffix}count"),
    ]


def expr_merge_mean_impl(column: str, suffix: str) -> list[pl.Expr]:
    count_all = pl.col(f"{column}{suffix}count").sum()
    mean_expr = (pl.col(f"{column}{suffix}") * pl.col(f"{column}{suffix}count")).sum() / count_all
    return [
        mean_expr.alias(f"{column}{suffix}"),
        count_all,
    ]


def expr_quantize_circ_mean_impl(column: str, low: float, high: float, suffix: str) -> list[pl.Expr]:
    angle = _to_angle(pl.col(column), low, high)
    angle_mean = pl.arctan2(angle.sin().mean(), angle.cos().mean())
    circ_mean = _from_angle(angle_mean, low, high)

    count = (pl.len() - pl.col(column).null_count()).alias(f"{column}{suffix}count")
    return [circ_mean.alias(f"{column}{suffix}"), count]


def expr_merge_circ_mean_impl(column: str, low: float, high: float, suffix: str) -> list[pl.Expr]:
    count = pl.col(f"{column}{suffix}count")
    count_all = pl.col(f"{column}{suffix}count").sum().alias(f"{column}{suffix}count")

    angle = _to_angle(pl.col(f"{column}{suffix}"), low, high)
    angle_mean = pl.arctan2((angle.sin() * count).sum(), (angle.cos() * count).sum())
    circ_mean = _from_angle(angle_mean, low, high)

    return [circ_mean.alias(f"{column}{suffix}"), count_all]


def _to_angle(x: pl.Expr, low: float, high: float) -> pl.Expr:
    range_val = high - low
    return 2 * math.pi * (x - low) / range_val


def _from_angle(angle: pl.Expr, low: float, high: float) -> pl.Expr:
    range_val = high - low

    normalized_angle = (angle + 2 * math.pi) % (2 * math.pi)
    x = (normalized_angle / (2 * math.pi)) * range_val + low

    return x


def expr_quantize_value_counts_impl(column: str, suffix: str) -> pl.Expr:
    return pl.col(column).value_counts(name=STRUCT_FIELD_COUNT).alias(f"{column}{suffix}")


def expr_merge_value_counts_impl(column: str, values: Iterable, suffix: str) -> pl.Expr:
    value_count_exprs = []
    x = pl.col(f"{column}{suffix}")
    for v in values:
        elem_field_expr = pl.element().struct.field(column)
        predicate_expr = elem_field_expr.eq(v) if v is not None else elem_field_expr.is_null()
        count_expr = (
            x
            .list.filter(predicate_expr)
            .explode(empty_as_null=False, keep_nulls=False)
            .struct.field(STRUCT_FIELD_COUNT).sum()
        )
        value_count_expr = pl.struct(pl.lit(v).alias(column), count_expr.alias(STRUCT_FIELD_COUNT))
        value_count_exprs.append(value_count_expr)

    return pl.concat_list(value_count_exprs).alias(f"{column}{suffix}")
