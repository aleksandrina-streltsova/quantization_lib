from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import Iterable

import polars as pl

from quantization._impl.aggregation import (
    expr_quantize_simple_impl, expr_merge_simple_impl,
    expr_quantize_off_impl, expr_merge_off_impl,
    expr_quantize_mean_impl, expr_merge_mean_impl,
    expr_quantize_circ_mean_impl, expr_merge_circ_mean_impl,
    expr_quantize_value_counts_impl, expr_merge_value_counts_impl
)


class Aggregation(ABC):
    """
    Interface for column aggregation to apply during:
      - Quantization (distillation within raw → quantized)
      - Merge (distillation across quantized frames)
    """

    def __init__(self):
        if not (self.suffix.startswith("__") and self.suffix.count("__") == 1):
            raise ValueError(f"Aggregation suffix must start with '__' and contain no other underscores, "
                             f"got '{self.suffix}'")

    @staticmethod
    def get_base_name(column_name: str) -> str | None:
        """
        Transforms aggregated column name to base column name or returns the provided column name without change.
        If the base name is empty, returns None.
        """
        if "__" not in column_name:
            return column_name

        base, suffix_part = column_name.rsplit("__", 1)

        if base == "":
            return None

        return base

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable unique name of the aggregation. Used for registry lookup.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def suffix(self) -> str:
        """
        Suffix appended to the original column name during aggregation.
        """
        raise NotImplementedError

    @abstractmethod
    def expr_quantize(self, column: str) -> pl.Expr | list[pl.Expr]:
        """
        Builds a Polars aggregation expression for this column during the quantize step.
        Must return Polars expressions compatible with:
          - group_by(...).agg(expr)
          - group_by(...).agg([expr1, expr2, ...])
        """
        raise NotImplementedError

    @abstractmethod
    def expr_merge(self, column: str) -> pl.Expr | list[pl.Expr]:
        """
        Builds a Polars aggregation expression for this column during the merge step.
        Must return Polars expressions compatible with:
          - group_by(...).agg(expr)
          - group_by(...).agg([expr1, expr2, ...])
        """
        raise NotImplementedError

    @abstractmethod
    def memory_factor(self, is_merge: bool) -> Number:
        """
        Estimated multiplicative memory factor for the QUANTIZE(MERGE) step.
        This is a heuristic for planning/estimating memory usage in large aggregations.
        """
        return 1.0

    def __repr__(self) -> str:
        return f"<Aggregation {self.name}>"


type AggregationKey = str | Aggregation


class AggOff(Aggregation):
    """
    Built-in aggregation that effectively turns aggregation off for a column.
    The values of the column are stored in lists, with no distillation.
    """

    @property
    def name(self) -> str:
        return "off"

    @property
    def suffix(self) -> str:
        return "__off"

    def expr_quantize(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_quantize_off_impl(column, self.suffix)

    def expr_merge(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_merge_off_impl(column, self.suffix)

    def memory_factor(self, is_merge: bool) -> Number:
        return 1.0


class AggMin(Aggregation):
    """
    Built-in min aggregation.
    """

    @property
    def name(self) -> str:
        return "min"

    @property
    def suffix(self) -> str:
        return "__min"

    def expr_quantize(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_quantize_simple_impl(column, "min", self.suffix)

    def expr_merge(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_merge_simple_impl(column, "min", self.suffix)

    def memory_factor(self, is_merge: bool) -> Number:
        return 1.0


class AggMax(Aggregation):
    """
    Built-in max aggregation.
    """

    @property
    def name(self) -> str:
        return "max"

    @property
    def suffix(self) -> str:
        return "__max"

    def expr_quantize(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_quantize_simple_impl(column, "max", self.suffix)

    def expr_merge(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_merge_simple_impl(column, "max", self.suffix)

    def memory_factor(self, is_merge: bool) -> Number:
        return 1.0


class AggMean(Aggregation):
    """
    Built-in mean aggregation.
    """

    @property
    def name(self) -> str:
        return "mean"

    @property
    def suffix(self) -> str:
        return "__mean"

    def expr_quantize(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_quantize_mean_impl(column, self.suffix)

    def expr_merge(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_merge_mean_impl(column, self.suffix)

    def memory_factor(self, is_merge: bool) -> Number:
        return 1.0


class AggCircularMean(Aggregation):
    """
    Circular mean aggregation for columns with a periodic range.
    """

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self.low = low
        self.high = high

    @property
    def range(self) -> Number:
        return self.high - self.low

    @property
    def name(self) -> str:
        return f"circ_mean_{self.low}_{self.high}"

    @property
    def suffix(self) -> str:
        return "__circmean"

    def expr_quantize(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_quantize_circ_mean_impl(column, self.low, self.high, self.suffix)

    def expr_merge(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_merge_circ_mean_impl(column, self.low, self.high, self.suffix)

    def memory_factor(self, is_merge: bool) -> Number:
        return 1.0


class AggValueCounts(Aggregation):
    """
    Aggregation that computes counts for specified values.
    The result is a list of structs, each containing a value and its count.
    """

    def __init__(self, values: Iterable):
        super().__init__()
        self.values = values

    @property
    def name(self) -> str:
        return f"value_counts_{'_'.join(self.values)}"

    @property
    def suffix(self) -> str:
        return "__valuecounts"

    def expr_quantize(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_quantize_value_counts_impl(column, self.suffix)

    def expr_merge(self, column: str) -> pl.Expr | list[pl.Expr]:
        return expr_merge_value_counts_impl(column, self.values, self.suffix)

    def memory_factor(self, is_merge: bool) -> Number:
        return 1.0


class AggregationRegistry:
    """
    Registry for named aggregations. Users can register custom aggregations implementing Aggregation.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, Aggregation] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for a in (AggOff(), AggMin(), AggMax(), AggMean()):
            self.register(a.name, a)

    def register(self, name: str, agg: Aggregation) -> None:
        """
        Registers an aggregation implementation under a given name.
        Overwrites any existing entry with the same name.
        """
        self._by_name[name] = agg

    def get(self, name: str) -> Aggregation | None:
        """
        Retrieves an aggregation by name or None if not registered.
        """
        return self._by_name.get(name, None)

    def __getitem__(self, item):
        try:
            return self._by_name[item]
        except KeyError:
            raise KeyError(f"{item!r} is not a known aggregation") from None

    def __iter__(self):
        return iter(self._by_name)


@dataclass(frozen=True)
class AggregationPlan:
    """
    Plan describing which aggregation to apply per column during:
      - Quantization (distillation within raw → quantized)
      - Merge (distillation across quantized frames)

    Notes:
      - Quantization columns are always included with their quantized values, but additional columns derived from
        the quantization columns can be created, as with any other column.

    Args:
        per_column: Mapping of column names to aggregation methods (name or instance).
        default_non_quant: Default aggregation methods used for columns that are not listed in :py:attr:`per_column` and
            are not quantization columns.
        default_quant: Default aggregation methods used for quantization columns that are not listed in
            :py:attr:`per_column`. If None, the quantized frame will include only the quantization columns with their
            quantized values. Otherwise, additional columns derived from the quantization columns will be created.
    """

    per_column: dict[str, AggregationKey | list[AggregationKey]]
    default_non_quant: AggregationKey | list[AggregationKey]
    default_quant: AggregationKey | list[AggregationKey] | None = None
