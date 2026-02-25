from numbers import Number

import numpy as np
import polars as pl
from numpy.typing import ArrayLike


def expr_round_fixed_edges_impl(column: str, edges: ArrayLike) -> pl.Expr:
    edges = np.asarray(edges)
    midpoints = (edges[:-1] + edges[1:]) / 2
    values_quant = np.pad(midpoints, (1, 1), mode="constant", constant_values=(edges[0], edges[-1]))

    breaks = (
        pl.col(column)
        .cut(breaks=edges, left_closed=True, include_breaks=True)
        .struct.field("breakpoint").alias(column)
    )

    values_old = np.pad(edges, (0, 1), mode="constant", constant_values=np.inf)

    return breaks.replace_strict(old=values_old, new=values_quant)


def expr_round_fixed_step_impl(column: str, step: Number, clip: tuple[Number, Number] | None,
                               use_integer_indices: bool = False) -> pl.Expr:
    expr_col = pl.col(column)

    if clip is not None:
        lower, upper = clip
        expr_col = expr_col.clip(lower, upper)

    expr = (expr_col / step).round()

    if use_integer_indices:
        return expr.cast(pl.Int32)

    return expr * step
