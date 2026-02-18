from abc import ABC, abstractmethod
from numbers import Number

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from quantization._impl.quant_columns import finalize_uncertainty_config_impl, finalize_probabilities_config_impl
from quantization._impl.rounding import expr_round_fixed_edges_impl, expr_round_fixed_step_impl
from quantization.types import FrameStatistics


class _BaseQuantColumnConfig(ABC):
    """
    Configuration for a single quantization column.

    Notes:
      - Configurations may be unfinalized if they need dataset-dependent computations
        (e.g., probabilities->edges, uncertainty->step).
      - Callers must first finalize configurations (see :py:func:~quantization.api.finalize_quant_column_configs())
        before quantize/merge/select.
    """
    def __init_subclass__(cls, *, _internal=False, **kwargs):
        # Block only direct subclassing of the private base; allow subclassing
        # of public intermediate bases (Finalized/Unfinalized).
        is_direct_base_subclass = _BaseQuantColumnConfig in cls.__bases__
        if is_direct_base_subclass and not _internal:
            raise TypeError("`_BaseQuantColumnConfig` cannot be subclassed externally. "
                            "Use one of [`FinalizedQuantColumnConfig`, `UnfinalizedQuantColumnConfig`] instead.")

    def __init__(self, name: str, clip: tuple[Number, Number] | None = None):
        self._name = name
        assert clip is None or len(clip) == 2, "Argument `clip` must be a tuple of two numbers."
        self._clip = clip

    @property
    def name(self) -> str:
        """
        Column name in :py:type:`~quantization.types.Frame`.
        """
        return self._name

    @property
    def clip(self) -> tuple[Number, Number] | None:
        """
        Optional clip range in case values should be clipped before quantization.
        """
        return self._clip

    def __repr__(self) -> str:
        return f"<Column config {self.name}>"


class FinalizedQuantColumnConfig(_BaseQuantColumnConfig, ABC, _internal=True):
    """
    Finalized configuration for a single quantization column.
    """

    @abstractmethod
    def expr_round(self) -> pl.Expr:
        """
        Builds a Polars expression for rounding/binning of this column.
        Must return Polars expressions compatible with :py:func:~polars.DataFrame.with_columns()
        """
        raise NotImplementedError


class UnfinalizedQuantColumnConfig(_BaseQuantColumnConfig, ABC, _internal=True):
    """
    Unfinalized configuration for a single quantization column.
    Must specify how to obtain the finalized configurations from the unfinalized ones.
    """

    @abstractmethod
    def finalize(self, frame_stats: FrameStatistics | None, *args, **kwargs) -> FinalizedQuantColumnConfig:
        """
        Finalizes configuration using provided frame statistics and/or any additional arguments
        required for data-dependent computations.
        """
        raise NotImplementedError


class FixedStepQuantColumnConfig(FinalizedQuantColumnConfig):
    """
    Finalized configuration with a fixed step size for rounding of a column.
    Rounded values constitute the nearest multiples of the provided step.
    """

    def __init__(self, name: str, step: Number, clip: tuple[Number, Number] | None = None,
                 use_integer_indices: bool = False):
        super().__init__(name, clip)
        self._step = step
        self._use_integer_indices = use_integer_indices

    @property
    def step(self):
        """
        Step size for rounding of a column.
        """
        return self._step

    @property
    def use_integer_indices(self):
        """
        Whether to use integer indices instead of the corresponding rounded values.
        """
        return self._use_integer_indices

    def expr_round(self) -> pl.Expr:
        return expr_round_fixed_step_impl(self._name, self._step, self._clip, self._use_integer_indices)


class FixedEdgesQuantColumnConfig(FinalizedQuantColumnConfig):
    """
    Finalized configuration with pre-computed bin edges for binning of a column.
    Rounded values constitute the midpoints of the corresponding bins.
    """

    def __init__(self, name: str, edges: ArrayLike, clip: tuple[Number, Number] | None = None) -> None:
        super().__init__(name, clip)
        assert len(edges) > 1, f"Argument `edges` must have at least 2 values, got {len(edges)}."

        if clip is not None:
            lower, upper = clip
            edges = [lower] + [edge for edge in edges if lower < edge < upper] + [upper]

        self._edges = np.array(sorted(edges), dtype=np.float64)
        self._clip = self._edges[0], self._edges[-1]

    @property
    def edges(self):
        """
        Edges for binning of a column.
        """
        return self._edges

    def expr_round(self) -> pl.Expr:
        return expr_round_fixed_edges_impl(self._name, self._edges)


class RoundingQuantColumnConfig(FinalizedQuantColumnConfig):
    """
    Finalized configuration with decimal precision for rounding of a column.
    """
    def __init__(self, name: str, decimals: int, clip: tuple[Number, Number] | None = None) -> None:
        super().__init__(name, clip)
        assert decimals >= 0, f"Argument `decimals` must be non-negative, got {decimals}."
        self._decimals = decimals

    @property
    def decimals(self) -> int:
        """
        Decimal precision for rounding.
        """
        return self._decimals

    def expr_round(self) -> pl.Expr:
        step = .1 ** self._decimals
        return expr_round_fixed_step_impl(self._name, step, self._clip)


class UncertaintyQuantColumnConfig(UnfinalizedQuantColumnConfig):
    """
    Unfinalized configuration with a known measurement uncertainty used to derive
    a fixed quantization step.

    Finalization estimates a scaling factor that achieves a target data reduction.
    A small, representative subset of the data is quantized repeatedly while
    adjusting the factor until the desired reduction characteristics are met.

    The finalized configuration is produced by converting each instance into a
    :py:class:`FixedStepQuantColumnConfig`, where:

        step = factor * uncertainty

    The resulting step is used for fixed-step rounding during quantization.
    """

    def __init__(self, name: str, uncertainty: Number, clip: tuple[Number, Number] | None = None) -> None:
        super().__init__(name, clip)
        self._uncertainty = uncertainty

    @property
    def uncertainty(self):
        """
        Measurement uncertainty used to derive a fixed quantization step.
        """
        return self._uncertainty

    def finalize(self, frame_stats: FrameStatistics | None, *args, **kwargs) -> FixedStepQuantColumnConfig:
        """
        This method overrides :py:meth:`UnfinalizedQuantColumnConfig.finalize`.
        """
        max_n_final = kwargs.get("max_n_final", None)
        if max_n_final is None:
            raise ValueError("`max_n_final` should be specified when finalizing `UncertaintyQuantColumnConfig`")

        return finalize_uncertainty_config_impl(self, frame_stats, max_n_final)


class ProbabilitiesQuantColumnConfig(UnfinalizedQuantColumnConfig):
    """
    Unfinalized configuration with target probabilities used to derive fixed bin
    edges.

    Finalization performs a data pass to estimate the empirical value distribution
    for the configured column. The specified probabilities are mapped onto that
    distribution to compute concrete bin edges.

    The finalized configuration is produced by converting each instance into a
    :py:class:`FixedEdgesQuantColumnConfig` using the calculated edges, which are
    subsequently used for deterministic binning.
    """

    def __init__(self, name: str, probabilities: ArrayLike, clip: tuple[Number, Number] | None = None) -> None:
        super().__init__(name, clip)
        self._probabilities = probabilities

    @property
    def probabilities(self):
        """
        Probabilities used to derive fixed bin edges.
        """
        return self._probabilities

    def finalize(self, frame_stats: FrameStatistics | None, *args, **kwargs) -> FixedEdgesQuantColumnConfig:
        """
        This method overrides :py:meth:`UnfinalizedQuantColumnConfig.finalize`.
        """
        return finalize_probabilities_config_impl(self, frame_stats)
