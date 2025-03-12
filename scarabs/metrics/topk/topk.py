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
"""Accuracy metric."""

import math

import datasets
import evaluate

_DESCRIPTION = """
recall metric

hit@K is the number of items that are predicted to be relevant and actually relevant.

ndcg@K is the normalized discounted cumulative gain.

mrr@K is the mean of reciprocal rank of the first relevant item.

"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `float`): Predicted labels.
    K (`int`): The number of items to consider for hit@K and ndcg@K. Defaults to 20.

Returns:
    hit@K ndcg@K mrr@K

Examples:

    >>> topk_metric = evaluate.load("topk")
    >>> results = topk_metric.compute(predictions=[0.9, 0.8, 0.5, 0.4, 0.2, 0.1], K=20)
    >>> print(results)
    {'hit@20': 1, 'ndcg@20': 1, 'mrr@20': 1}

"""


_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={merlin},
}
"""


def getHitRatio(ranklist):
    for item, score in ranklist:
        if item == 0:
            return 1
    return 0


def getNDCG(ranklist):
    for i in range(len(ranklist)):
        item = ranklist[i][0]
        if item == 0:
            return math.log(2) / math.log(i + 2)
    return 0


def getMRR(ranklist):
    for i in range(len(ranklist)):
        item = ranklist[i][0]
        if item == 0:
            return 1 / (i + 1)
    return 0


def split_list(lst, chunk_size=100):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Topk(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float32")),
                    "references": datasets.Sequence(datasets.Value("float32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("float32"),
                }
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"
            ],
        )

    def _compute(self, predictions=None, references=None, K=20, candidate_num=100):
        smaller_lists = split_list(predictions, chunk_size=candidate_num)
        nums = list(range(candidate_num))

        mrr = 0
        ndcg = 0
        hr = 0

        for i, small_list in enumerate(smaller_lists):
            if len(small_list) != candidate_num:
                continue

            small_list = small_list[::-1]
            nums = nums[::-1]

            pred_list = zip(nums, small_list)
            pred_list = sorted(pred_list, key=lambda x: x[1], reverse=True)[:K]
            hr += getHitRatio(pred_list)
            ndcg += getNDCG(pred_list)
            mrr += getMRR(pred_list)

        mrr = mrr / len(smaller_lists)
        hr = hr / len(smaller_lists)
        ndcg = ndcg / len(smaller_lists)
        return {
            f"mrr@{K}": mrr,
            f"hit@{K}": hr,
            f"ndcg@{K}": ndcg,
        }
