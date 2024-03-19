"""
Test the custom metric classes for model training with PyTorch Ignite.
"""
# pylint: disable=missing-function-docstring, protected-access

import pytest
import torch
from ignite.exceptions import NotComputableError
from sklearn.metrics import f1_score

from ctgnn.training.metrics import F1
from ctgnn.utils import to_tensor


def test_reset():
    f1 = F1()
    f1._true_positives = 5
    f1._false_positives = 3
    f1._false_negatives = 2
    f1._weight = 1
    f1._updated = True

    f1.reset()

    assert f1._true_positives == 0
    assert f1._false_positives == 0
    assert f1._false_negatives == 0
    assert f1._weight == 0
    assert not f1._updated


def test_update_micro():
    f1 = F1(average="micro")
    y_pred = to_tensor(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.05, 0.9, 0.05, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.7, 0.1, 0.2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    y = to_tensor([0, 1, 1, 1, 3])

    f1.update((y_pred, y))

    # micro checks each class separately and adds them up
    assert f1._true_positives == to_tensor(3)
    assert f1._false_positives == to_tensor(2)
    assert f1._false_negatives == to_tensor(2)


def test_update_false():
    f1 = F1(average=False)
    y_pred = to_tensor(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.05, 0.9, 0.05, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.7, 0.1, 0.2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    y = to_tensor([0, 1, 1, 1, 3])

    f1.update((y_pred, y))

    # micro checks each class separately and adds them up
    assert (f1._true_positives == to_tensor([1, 1, 0, 1])).all()
    assert (f1._false_positives == to_tensor([1, 0, 1, 0])).all()
    assert (f1._false_negatives == to_tensor([0, 2, 0, 0])).all()


def test_update_weighted():
    f1 = F1(average="weighted")
    y_pred = to_tensor(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.05, 0.9, 0.05, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.7, 0.1, 0.2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    y = to_tensor([0, 1, 1, 1, 3])

    f1.update((y_pred, y))

    # micro checks each class separately and adds them up
    assert (f1._true_positives == to_tensor([1, 1, 0, 1])).all()
    assert (f1._false_positives == to_tensor([1, 0, 1, 0])).all()
    assert (f1._false_negatives == to_tensor([0, 2, 0, 0])).all()
    assert (f1._weight == to_tensor([1, 3, 0, 1])).all()


def test_compute_micro():
    tp = to_tensor(3.0)
    fp = to_tensor(2.0)
    fn = to_tensor(2.0)

    f1 = F1(average="micro")
    f1._true_positives = tp
    f1._false_positives = fp
    f1._false_negatives = fn
    f1._updated = True

    numerator = 2 * tp
    denominator = numerator + fp + fn
    score = numerator / denominator
    # Is not exact because of epsilon to avoid division by zero
    assert to_tensor(f1.compute()).isclose(score)


def test_compute_weighted():
    tp = to_tensor([1.0, 1.0, 0.0, 1.0])
    fp = to_tensor([1.0, 0.0, 1.0, 0.0])
    fn = to_tensor([0.0, 2.0, 0.0, 0.0])
    weight = to_tensor([1.0, 3.0, 0.0, 1.0])

    f1 = F1(average="weighted")
    f1._true_positives = tp
    f1._false_positives = fp
    f1._false_negatives = fn
    f1._weight = weight
    f1._updated = True

    numerator = 2 * tp
    denominator = numerator + fp + fn
    score = numerator / denominator
    weighted_score = (score * weight).sum() / weight.sum()
    # Is not exact because of epsilon to avoid division by zero
    assert to_tensor(f1.compute()).isclose(weighted_score)


def test_compute_false():
    tp = to_tensor([1.0, 1.0, 0.0, 1.0])
    fp = to_tensor([1.0, 0.0, 1.0, 0.0])
    fn = to_tensor([0.0, 2.0, 0.0, 0.0])

    f1 = F1(average=False)
    f1._true_positives = tp
    f1._false_positives = fp
    f1._false_negatives = fn
    f1._updated = True

    numerator = 2 * tp
    denominator = numerator + fp + fn
    score = numerator / denominator
    assert (to_tensor(f1.compute()) == score).all()


def test_compute_macro():
    tp = to_tensor([1.0, 1.0, 0.0, 1.0])
    fp = to_tensor([1.0, 0.0, 1.0, 0.0])
    fn = to_tensor([0.0, 2.0, 0.0, 0.0])

    f1 = F1(average="macro")
    f1._true_positives = tp
    f1._false_positives = fp
    f1._false_negatives = fn
    f1._updated = True

    numerator = 2 * tp
    denominator = numerator + fp + fn
    macro_score = (numerator / denominator).mean()
    assert to_tensor(f1.compute()).isclose(macro_score)


def test_check_f1_with_scikit_implementation():
    y = torch.randint(0, 10, (100,))
    y_pred = torch.softmax(torch.rand((100, 10)), dim=1)

    f1 = F1(average="micro")
    f1.update((y_pred, y))
    score_micro = f1.compute()
    scikit_score_micro = f1_score(y, y_pred.argmax(dim=1), average="micro")
    assert to_tensor(score_micro, dtype=torch.float).isclose(
        to_tensor(scikit_score_micro, dtype=torch.float)
    )

    f1 = F1(average="macro")
    f1.update((y_pred, y))
    score_macro = f1.compute()
    scikit_score_macro = f1_score(y, y_pred.argmax(dim=1), average="macro")
    assert to_tensor(score_macro, dtype=torch.float).isclose(
        to_tensor(scikit_score_macro, dtype=torch.float)
    )

    f1 = F1(average="weighted")
    f1.update((y_pred, y))
    score_weighted = f1.compute()
    scikit_score_weighted = f1_score(y, y_pred.argmax(dim=1), average="weighted")
    assert to_tensor(score_weighted, dtype=torch.float).isclose(
        to_tensor(scikit_score_weighted, dtype=torch.float)
    )

    f1 = F1(average=False)
    f1.update((y_pred, y))
    score_false = f1.compute()
    scikit_score_false = f1_score(y, y_pred.argmax(dim=1), average=None)
    assert (
        to_tensor(score_false, dtype=torch.float)
        == to_tensor(scikit_score_false, dtype=torch.float)
    ).all()


def test_f1_compute_fails():
    f1 = F1()
    with pytest.raises(NotComputableError):
        f1.compute()
