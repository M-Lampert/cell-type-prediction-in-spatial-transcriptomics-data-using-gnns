"""
Module for computing useful things for the results.
E.g. averaging over the results of multiple runs.
"""
from pathlib import Path

import pandas as pd

__all__ = [
    "average_results",
    "get_best_results_per",
]

cols_to_avg = [
    "runtime",
    "train_loss",
    "train_accuracy",
    "train_f1",
    "eval_loss",
    "eval_accuracy",
    "eval_f1",
    "test_loss",
    "test_accuracy",
    "test_f1",
]
root = Path(__file__).parents[3]


def average_results(
    results: pd.DataFrame, avg_cols: list | None = None, ref_col: str = "test_f1"
) -> pd.DataFrame:
    """
    Averages the results of multiple runs.
    Uses the `output_dir` of the best run according to ref_col.

    Args:
        results: The results to average.
        avg_cols: The columns to average.
            Defaults to loss, f1 and accuracy for train, eval and test dataset.
        ref_col: The column to use for finding the best result.
            Defaults to test_f1.
    """
    if avg_cols is None:
        avg_cols = cols_to_avg

    results = get_groupable_df(results)

    all_cols = list(results.columns)
    avg_cols = [c for c in avg_cols if c in all_cols]
    assert ref_col in avg_cols, f"ref_col {ref_col} not in avg_cols {avg_cols}"

    group_cols = get_group_cols(all_cols, avg_cols)
    idx_max_results = results.groupby(group_cols)[ref_col].idxmax()
    max_output_dirs = results.loc[idx_max_results, group_cols + ["output_dir"]]

    avg_results = results.groupby(group_cols)[avg_cols].mean().reset_index()
    avg_results = pd.merge(avg_results, max_output_dirs, on=group_cols)

    return avg_results


def get_best_results_per(
    results: pd.DataFrame,
    group_cols: list[str] | None = None,
    ref_col: str = "test_f1",
) -> pd.DataFrame:
    """
    Return all runs of the best result gouped by `group_cols` averaged over all runs.

    Args:
        results: The results to average.
        group_cols: The columns to group by.
            Defaults to no grouping i.e. the best overall result.
        ref_col: The column to use for finding the best result.
    """
    results = get_groupable_df(results)

    # Find the best average result for each group.
    avg_results = average_results(results, ref_col=ref_col)
    if group_cols is None:
        # Make sure that we get a dataframe in the end and not a series.
        idx_best_results = [avg_results[ref_col].idxmax()]
    else:
        idx_best_results = avg_results.groupby(group_cols)[ref_col].idxmax()
    all_group_cols = get_group_cols(list(results.columns))
    best_avg_results = avg_results.loc[idx_best_results, all_group_cols]

    # Get all runs of the best result.
    best_results = pd.merge(best_avg_results, results, on=all_group_cols, how="inner")
    return best_results


def get_group_cols(all_cols: list[str], avg_cols: list[str] | None = None) -> list[str]:
    """
    Return the columns that are used for grouping.

    Args:
        all_cols: All columns of the results.
        avg_cols: The columns to average.
            Defaults to loss, f1 and accuracy for train, eval and test dataset.
    """
    if avg_cols is None:
        avg_cols = cols_to_avg

    group_cols = [
        c for c in all_cols if (c not in avg_cols) and (c not in ["output_dir", "runs"])
    ]
    return group_cols


def get_groupable_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe that can be grouped without errors.
    """
    # Set null values to symbolic string because it makes problems when grouping.
    for col in cols_to_avg:
        if col in df.columns:
            df.loc[:, col] = df.loc[:, col].fillna(-1)
    df = df.fillna("none")

    # A list inside the dataframe brings problems when grouping.
    # Therefore, we convert the list to a string.
    # https://stackoverflow.com/questions/45306988/column-of-lists-convert-list-to-string-as-a-new-column
    # Question by: https://stackoverflow.com/users/3885217/clg4
    # Edited by: https://stackoverflow.com/users/11107541/starball
    # Answered by: https://stackoverflow.com/users/4909087/cs95
    if not isinstance(df.loc[0, "splits"], str):
        df["splits"] = [",".join(map(str, split)) for split in df["splits"]]

    return df
