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


def getRecall(pred_item, true_item, k):
    pred_item = pred_item[:k]
    hit_item = set(pred_item) & set(true_item)
    recall = len(hit_item) / (len(true_item) + 1e-12)
    return recall


def getNRecall(pred_item, true_item, k):
    pred_item = pred_item[:k]
    hit_item = set(pred_item) & set(true_item)
    recall = len(hit_item) / min(k, len(pred_item) + 1e-12)
    return recall


def getPrecision(pred_item, true_item, k):
    pred_item = pred_item[:k]
    hit_item = set(pred_item) & set(true_item)
    precision = len(hit_item) / (k + 1e-12)
    return precision


def getF1(pred_item, true_item, k):
    p = getPrecision(pred_item, true_item, k)
    r = getRecall(pred_item, true_item, k)
    f1 = 2 * p * r / (p + r + 1e-12)
    return f1


def getDCG(pred_item, true_item, k):
    pred_item = pred_item[:k]
    true_item = set(true_item)
    dcg = 0
    for i, item in enumerate(pred_item):
        if item in true_item:
            dcg += 1 / math.log(2 + i)
    return dcg


def getNDCG(pred_item, true_item, k):
    pred_item = pred_item[:k]
    idcg = getDCG(true_item, true_item, k)
    dcg = getDCG(pred_item, true_item, k)
    return dcg / (idcg + 1e-12)


def getMRR(pred_item, true_item, k):
    pred_item = pred_item[:k]
    true_item = set(true_item)
    mrr = 0
    for i, item in enumerate(pred_item):
        if item in true_item:
            mrr += 1 / (i + 1.0)
    return mrr


def getHitRatio(pred_item, true_item, k):
    pred_item = pred_item[:k]
    hit_items = set(pred_item) & set(true_item)
    hit_rate = 1 if len(hit_items) > 0 else 0
    return hit_rate


def getMAP(pred_item, true_item, k):
    pred_item = pred_item[:k]
    true_item = set(true_item)
    pos = 0
    precision = 0
    for i, item in enumerate(pred_item):
        if item in true_item:
            pos += 1
            precision += pos / (i + 1.0)
    return precision / (pos + 1e-12)


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Topk_recall(evaluate.Metric):
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
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"
            ],
        )

    def _compute(self, predictions=None, references=None, K=20):
        assert predictions is not None
        assert references is not None
        assert len(predictions) == len(references)
        Recall = 0
        NRecall = 0
        Precision = 0
        F1 = 0
        NDCG = 0
        MRR = 0
        HR = 0
        MAP = 0

        for i in range(len(predictions)):
            Recall += getRecall(predictions[i], references[i], K)
            NRecall += getNRecall(predictions[i], references[i], K)
            Precision += getPrecision(predictions[i], references[i], K)
            F1 += getF1(predictions[i], references[i], K)
            NDCG += getNDCG(predictions[i], references[i], K)
            MRR += getMRR(predictions[i], references[i], K)
            HR += getHitRatio(predictions[i], references[i], K)
            MAP += getMAP(predictions[i], references[i], K)

        Recall /= len(predictions)
        NRecall /= len(predictions)
        Precision /= len(predictions)
        F1 /= len(predictions)
        NDCG /= len(predictions)
        MRR /= len(predictions)
        HR /= len(predictions)
        MAP /= len(predictions)

        return {
            f"Recall@{K}": Recall,
            f"NRecall@{K}": NRecall,
            f"Precision@{K}": Precision,
            f"F1@{K}": F1,
            f"NDCG@{K}": NDCG,
            f"MRR@{K}": MRR,
            f"HR@{K}": HR,
            f"MAP@{K}": MAP,
        }
