from typing import Any, Callable

import polars as pl
import polars.testing

from quantization.types import Frame


def assert_frames_equal(actual: pl.DataFrame, expected: pl.DataFrame):
    # Ensure columns are in the same order for comparison
    actual = actual.select(sorted(actual.columns))
    expected = expected.select(sorted(expected.columns))

    # For columns that are lists of structs (like value_counts), sort the lists to ensure deterministic comparison
    for col in actual.columns:
        if isinstance(actual.schema[col], pl.List) and isinstance(actual.schema[col].inner, pl.Struct):
            # Sorting list of structs by all fields
            actual = actual.with_columns(pl.col(col).list.sort())
            expected = expected.with_columns(pl.col(col).list.sort())

    polars.testing.assert_frame_equal(actual, expected, check_dtypes=False)


def run_test_twice(
    test_func: Callable[[Frame], Frame],
    input_data: dict[str, Any] | list[dict[str, Any]],
    expected_data: dict[str, Any],
):
    """
    Runs the test function twice: once with a DataFrame and once with a LazyFrame.
    """
    is_multi_frame = isinstance(input_data, list)

    expected_df = pl.DataFrame(expected_data, strict=False)
    expected_df = expected_df.sort(expected_df.columns)

    for frame_constructor in [pl.DataFrame, pl.LazyFrame]:
        if is_multi_frame:
            frames = [frame_constructor(d, strict=False) for d in input_data]
            result = test_func(frames)
        else:
            df = frame_constructor(input_data, strict=False)
            result = test_func(df)

        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        result = result.sort(result.columns)
        assert_frames_equal(result, expected_df)
