# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""F1 metric."""

import datasets
from sklearn.metrics import hamming_loss

import evaluate


_DESCRIPTION = """
he Hamming loss is the fraction of labels that are incorrectly predicted.:
n multiclass classification, the Hamming loss corresponds to the Hamming distance between y_true and y_pred which is equivalent to the subset zero_one_loss function, when normalize parameter is set to True.

In multilabel classification, the Hamming loss is different from the subset zero-one loss. The zero-one loss considers the entire set of labels for a given sample incorrect if it does not entirely match the true set of labels. Hamming loss is more forgiving in that it penalizes only the individual labels.

The Hamming loss is upperbounded by the subset zero-one loss, when normalize parameter is set to True. It is always between 0 and 1, lower being better.
"""


_KWARGS_DESCRIPTION = """
Parameters:
y_true： 1d array-like, or label indicator array / sparse matrix
Ground truth (correct) labels.

y_pred：1d array-like, or label indicator array / sparse matrix
Predicted labels, as returned by a classifier.

sample_weight：array-like of shape (n_samples,), default=None
Sample weights.

Returns:
    loss：float or int
        Return the average Hamming loss between element of y_true and y_pred.

Examples:

    Example 1-A simple binary example
        >>> from sklearn.metrics import hamming_loss
        >>> y_pred = [1, 2, 3, 4]
        >>> y_true = [2, 2, 3, 4]
        >>> hamming_loss(y_true, y_pred)
        0.25

    In the multilabel case with binary label indicators:
        >>> import numpy as np
        >>> hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))
        0.75
"""


_CITATION = """
@article{scikit-learn,
    title={Scikit-learn: Machine Learning in {P}ython},
    author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
           and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
           and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
           Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
    journal={Journal of Machine Learning Research},
    volume={12},
    pages={2825--2830},
    year={2011}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class HammingLoss(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html"
            ],
        )

    def _compute(
        self,
        predictions,
        references,
        sample_weight=None,
    ):
        score = hamming_loss(
            references,
            predictions,
            sample_weight=sample_weight,
        )
        return {"hammingloss": float(score)}
