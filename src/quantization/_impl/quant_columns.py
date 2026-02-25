from __future__ import annotations

import logging

import quantization.quant_columns as quant_columns
from quantization.types import FrameStatistics


def finalize_uncertainty_config_impl(config: quant_columns.UncertaintyQuantColumnConfig,
                                     frame_stats: FrameStatistics,
                                     max_n_final: int) -> quant_columns.FixedStepQuantColumnConfig:
    factors = [f for f, size in frame_stats.factor2size.items() if size <= max_n_final]
    if factors:
        best_factor = min(factors)
    else:
        best_factor = max(frame_stats.factor2size.keys())
        logging.warning("No suitable factor found for column `%s` with max_n_final=%d; "
                        "using largest available factor=%s.", config.name, max_n_final, best_factor)

    return quant_columns.FixedStepQuantColumnConfig(config.name, best_factor * config.uncertainty, config.clip)


def finalize_probabilities_config_impl(config: quant_columns.ProbabilitiesQuantColumnConfig,
                                       frame_stats: FrameStatistics) -> quant_columns.FixedEdgesQuantColumnConfig:

    ecdf = frame_stats.column2ecdf[config.name]
    edges = [ecdf.quantile(q) for q in config.probabilities]

    return quant_columns.FixedEdgesQuantColumnConfig(config.name, edges=edges, clip=config.clip)
