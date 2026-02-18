import pathlib
from dataclasses import dataclass
from typing import Iterable

from quantization.aggregation import AggregationPlan, AggregationRegistry
from quantization.constants import COLUMN_SUFFIX_QUANT
from quantization.quant_columns import FinalizedQuantColumnConfig


@dataclass(frozen=True)
class QuantizationConfig:
    """
    Top-level configuration for quantize/merge/select steps.

    Args:
    	quant_column_configs: User-specified configurations for quantization columns.
    	    These must be finalized before running steps.
    	agg_plan: Aggregation plan used for distillation during quantize and merge steps.
    	output_dir: Directory for intermediate and final data files.
    """
    quant_column_configs: Iterable[FinalizedQuantColumnConfig]
    agg_plan: AggregationPlan
    output_dir: str | pathlib.Path
    agg_registry: AggregationRegistry = AggregationRegistry()

    def __repr__(self) -> str:
        return f"<Config for {self.output_dir}>"

    def quant_columns(self, with_suffix=False) -> list[str]:
        if with_suffix:
            return [f"{col_config.name}{COLUMN_SUFFIX_QUANT}" for col_config in self.quant_column_configs]

        return [col_config.name for col_config in self.quant_column_configs]
