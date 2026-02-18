from dataclasses import dataclass
from enum import Enum

import polars as pl

type Frame = pl.DataFrame | pl.LazyFrame


@dataclass(frozen=True)
class FrameStatistics:
    column2ecdf: dict[str, pl.Series]
    factor2size: dict[float, int]


class QuantizationStep(Enum):
    FINALIZE = "finalize"
    QUANTIZE = "quantize"
    MERGE = "merge"
    SELECT = "select"

    def __repr__(self) -> str:
        return f"<Step {self.name}>"
