"""
This module contains the implementation for the F1 metric to be used in PyTorch Ignite.
It is based on the implementation of the Recall and Precision metrics from PyTorch
Ignite. See https://pytorch.org/ignite/v0.4.13/_modules/ignite/metrics/recall.html.
"""

from typing import Sequence, Union, cast

import ignite.distributed as idist
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.metrics.precision import _BasePrecisionRecall


class F1(_BasePrecisionRecall):
    r"""Calculates F1 for binary, multiclass and multilabel data.

    $\text{F1} = \frac{ 2TP }{ 2TP + FP + FN }$

    where $\text{TP}$ is true positives, $\text{FN}$ is false negatives
    and $\text{FP}$ is false positives.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
        or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).

    Args:
        output_transform: a callable that is used to transform the
            `ignite.engine.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you
            have a multi-output model and you want to compute the metric with
            respect to one of the outputs.
        average: available options are

            False
              default option. For multicalss and multilabel inputs, per class
              and per label metric is returned respectively.

            None
              like `False` option except that per class metric is returned
              for binary data as well. For compatibility with Scikit-Learn api.

            'micro'
              Metric is computed counting stats of classes/labels altogether.

            'weighted'
              like macro F1 but considers class/label imbalance. For binary and
              multiclass input, it computes metric for each class then returns
              average of them weighted by support of classes (number of actual
              samples in each class). For multilabel input, it computes F1 for
              each label then returns average of them weighted by support
              of labels (number of actual positive samples in each label).

            macro
              computes macro F1 which is unweighted average of metric computed across
              classes or labels.

            True
              like macro option. For backward compatibility.
        is_multilabel: flag to use in multilabel case. By default, value is False.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update``
            method is non-blocking. By default, CPU.
    """
    _state_dict_all_req_keys = (
        "_true_positives",
        "_false_positives",
        "_false_negatives",
        "_weight",
        "_updated",
    )  # type: ignore[assignment]

    @reinit__is_reduced
    def reset(self) -> None:
        self._true_positives: Union[int, torch.Tensor] = 0
        self._false_positives: Union[int, torch.Tensor] = 0
        self._false_negatives: Union[int, torch.Tensor] = 0
        self._weight: Union[int, torch.Tensor] = 0
        self._updated = False

        super(_BasePrecisionRecall, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y, correct = self._prepare_output(output)

        if self._average == "micro":
            self._true_positives += correct.sum()
            self._false_positives += y_pred.sum() - correct.sum()
            self._false_negatives += y.sum() - correct.sum()
        else:  # _average in [False, 'macro', 'weighted']
            self._true_positives += correct.sum(dim=0)
            self._false_positives += y_pred.sum(dim=0) - correct.sum(dim=0)
            self._false_negatives += y.sum(dim=0) - correct.sum(dim=0)

            if self._average == "weighted":
                self._weight += y.sum(dim=0)

        self._updated = True

    @sync_all_reduce("_true_positives", "_false_positives", "_false_negatives")
    def compute(self) -> Union[torch.Tensor, float]:
        r"""
        Return value of the metric for `average` options `'weighted'`
        and `'macro'` is computed as follows.
            $\text{F1} = \frac{ 2TP }{ 2TP + FP + FN } \cdot weight$
        wherein `weight` is the internal variable `_weight` for `'weighted'`
        option and `1/C` for the `macro` one. `C` is the number of classes/labels.

        Return value of the metric for `average` options `'micro'`, `False` and
        None is as follows.
            $\text{F1} = \frac{ 2TP }{ 2TP + FP + FN }$
        """

        if not self._updated:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one \
                example before it can be computed."
            )

        numerator = 2.0 * self._true_positives
        denominator = numerator + self._false_positives + self._false_negatives
        fraction = numerator / (denominator + self.eps)

        if self._average == "weighted":
            _weight = idist.all_reduce(self._weight.clone())  # type: ignore[union-attr]
            sum_of_weights = cast(torch.Tensor, _weight).sum() + self.eps
            return ((fraction @ _weight) / sum_of_weights).item()  # type: ignore
        if self._average == "micro":
            return cast(torch.Tensor, fraction).item()
        if self._average == "macro":
            return cast(torch.Tensor, fraction).mean().item()
        return fraction
