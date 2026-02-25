import logging
from typing import Callable

import polars as pl

from quantization.constants import COLUMN_COUNT
from quantization.types import Frame, QuantizationStep


class LogCallback:
    """
    Callback for Polars' `inspect` method or direct invocation on a DataFrame.
    Can be used to log specific metrics (like row counts) or perform assertions.
    """

    def __init__(
        self,
        message: str,
        extractor: Callable[[pl.DataFrame], int] = lambda df: df.height,
        on_value: Callable[[int], None] = None
    ):
        self.message = message
        self.extractor = extractor
        self.on_value = on_value

    def format(self, df: pl.DataFrame) -> str:
        value = self.extractor(df)
        if self.on_value:
            self.on_value(value)
        logging.info(self.message, value)
        return ""


def inspect_or_apply(frame: Frame, callback: LogCallback) -> Frame:
    """
    Utility to apply a callback to a frame.
    If the frame is a LazyFrame, it uses `.inspect(callback)`.
    If the frame is a DataFrame, it invokes the callback directly.
    """
    if isinstance(frame, pl.LazyFrame):
        return frame.inspect(callback)
    elif isinstance(frame, pl.DataFrame):
        callback.format(frame)
        return frame
    else:
        raise TypeError(f"Expected pl.DataFrame or pl.LazyFrame, got {type(frame)}")


class _QuantizationLogger:
    def __init__(self, step: QuantizationStep):
        self.total_count_before = [0]
        self.step_name = step.value

    def log_before(self, frame: Frame) -> Frame:
        def set_before(v_before):
            self.total_count_before[0] = v_before

        msg = f"Rows before {self.step_name}: %d"
        extractor = self._get_total_count if self.step_name == QuantizationStep.MERGE.value else lambda df: df.height
        return inspect_or_apply(frame, LogCallback(msg, extractor=extractor, on_value=set_before))

    def log_after(self, frame: Frame) -> Frame:
        frame = inspect_or_apply(frame, LogCallback(f"Rows after {self.step_name}: %d"))

        def check_after(v_after):
            v_before = self.total_count_before[0]
            assert v_before == v_after, f"Total count changed after {self.step_name}: {v_after} != {v_before}."

        return inspect_or_apply(
            frame,
            LogCallback("Total count: %d", extractor=self._get_total_count, on_value=check_after),
        )

    @staticmethod
    def _get_total_count(df: pl.DataFrame) -> int:
        return df.get_column(COLUMN_COUNT).cast(pl.Int64).sum()
