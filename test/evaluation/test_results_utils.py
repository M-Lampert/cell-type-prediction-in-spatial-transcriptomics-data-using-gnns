"""
Tests the utils used for analyzing the model results and
getting the best results and models.
"""
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access

import pandas as pd
import pytest

from ctgnn.evaluation.results_utils import (
    average_results,
    get_best_results_per,
    get_group_cols,
    get_groupable_df,
)


@pytest.fixture
def example_df():
    return pd.DataFrame(
        {
            "train_loss": list(range(1, 9))[::-1],
            "test_loss": list(range(3, 11))[::-1],
            "train_f1": list(range(5, 13)),
            "test_f1": list(range(7, 15)),
            "runs": [1, 2, 1, 2, 1, 2, 1, 2],
            "output_dir": [
                "foo_1",
                "foo_2",
                "bar_1",
                "bar_2",
                "baz_1",
                "baz_2",
                "qux_1",
                "qux_2",
            ],
            "seed": [1, 1, 1, 1, 2, 2, 2, 2],
            "other_param": [1, 1, 2, 2, 1, 1, 2, 2],
            "splits": [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
        }
    )


def test_average_results(example_df):
    avg_cols = ["train_loss", "test_loss", "train_f1", "test_f1"]
    avg_df = average_results(example_df, avg_cols)
    assert avg_df.shape == (4, 8)
    assert avg_df.loc[0, "train_loss"] == 7.5
    assert avg_df.loc[1, "test_loss"] == 7.5
    assert avg_df.loc[2, "train_f1"] == 9.5
    assert avg_df.loc[3, "test_f1"] == 13.5
    assert avg_df.loc[:, "output_dir"].tolist() == ["foo_2", "bar_2", "baz_2", "qux_2"]
    assert "runs" not in avg_df.columns

    avg_df = average_results(example_df, ref_col="test_loss")
    assert avg_df.loc[:, "output_dir"].tolist() == ["foo_1", "bar_1", "baz_1", "qux_1"]


def test_get_best_results_per(example_df):
    grouped_best_df = get_best_results_per(example_df, ["seed"])
    assert grouped_best_df.shape == (4, 9)
    assert grouped_best_df.loc[0, "train_loss"] == 6
    assert grouped_best_df.loc[1, "train_loss"] == 5
    assert grouped_best_df.loc[2, "test_f1"] == 13
    assert grouped_best_df.loc[3, "test_f1"] == 14
    assert grouped_best_df.loc[:, "output_dir"].tolist() == [
        "bar_1",
        "bar_2",
        "qux_1",
        "qux_2",
    ]

    grouped_best_df = get_best_results_per(example_df, ["seed", "other_param"])
    assert grouped_best_df.shape == (8, 9)

    grouped_best_df = get_best_results_per(example_df, ref_col="test_loss")
    assert grouped_best_df.shape == (2, 9)
    assert grouped_best_df.loc[0, "test_loss"] == 10
    assert grouped_best_df.loc[1, "test_loss"] == 9
    assert grouped_best_df.loc[:, "output_dir"].tolist() == ["foo_1", "foo_2"]


def test_get_group_cols():
    avg_cols = ["foo", "bar"]
    all_cols = ["foo", "bar", "baz", "runs", "output_dir"]
    group_cols = get_group_cols(all_cols, avg_cols)
    assert group_cols == ["baz"]

    all_cols = ["test_loss", "train_f1", "baz", "runs", "output_dir", "seed"]
    group_cols = get_group_cols(all_cols)
    assert group_cols == ["baz", "seed"]


def test_get_groupable_df():
    df = pd.DataFrame(
        {
            "foo": [None, 2, 3],
            "splits": [[4, 2], [5, 3], [6, 4]],
        }
    )
    assert df.loc[:, "foo"].isnull().sum() == 1
    assert df.loc[:, "splits"].apply(lambda x: isinstance(x, list)).sum() == 3
    groupable_df = get_groupable_df(df)

    # check for None values
    assert groupable_df.loc[:, "foo"].isnull().sum() == 0
    # check for list values
    assert groupable_df.loc[:, "splits"].apply(lambda x: isinstance(x, list)).sum() == 0
