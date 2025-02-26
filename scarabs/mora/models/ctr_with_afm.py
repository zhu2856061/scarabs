# -*- coding: utf-8 -*-
# @Time   : 2024/07/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class CtrModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CtrWithAFMConfig(PretrainedConfig):
    model_type = "CtrWithAFM"

    def __init__(
        self,
        features=None,
        label_name="label",
        hidden_dim=8,
        attention_dim=8,
        attention_dropout=[0, 0],
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attention_dropout = attention_dropout
        self.regularizer = regularizer


class EmbeddingUnitLayer(torch.nn.Module):
    def __init__(self, features_size, embedding_size) -> None:
        super(EmbeddingUnitLayer, self).__init__()

        self._embedding = torch.nn.Embedding(features_size, embedding_size)

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.normal_(self._embedding.weight, std=1e-4)

    def forward(self, X, meaning=True):
        _tmp = self._embedding(X)
        if meaning:
            return torch.mean(_tmp, dim=1)
        else:
            return _tmp


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, features, embedding_size) -> None:
        super().__init__()

        self.features = features

        self.feature_embedding = torch.nn.ModuleDict()
        for name, feature in self.features.items():
            if feature["shared_embed_name"] is not None:
                continue

            self.feature_embedding[name] = EmbeddingUnitLayer(
                len(feature["vocab"]), embedding_size
            )

    def forward(self, **kwargs):
        _inputs = {}
        for name, feature in self.features.items():
            if feature["shared_embed_name"] is None:  # shared_embed
                _tmp = self.feature_embedding[name](kwargs[name], meaning=True)

            else:
                _tmp = self.feature_embedding[feature["shared_embed_name"]](
                    kwargs[name], meaning=True
                )
            _inputs[name] = _tmp

        return _inputs


class LogitisticLayer(torch.nn.Module):
    def __init__(self, features) -> None:
        super().__init__()

        self.features = features

        self.feature_w = torch.nn.ModuleDict()
        for name, feature in self.features.items():
            if feature["shared_embed_name"] is not None:
                continue
            self.feature_w[name] = EmbeddingUnitLayer(len(feature["vocab"]), 1)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1))

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.zeros_(self.bias)

    def forward(self, **kwargs):
        _inputs = []
        for name, feature in self.features.items():
            if feature["shared_embed_name"] is None:  # shared_embed
                _tmp = self.feature_w[name](kwargs[name], meaning=True)
            else:
                _tmp = self.feature_w[feature["shared_embed_name"]](
                    kwargs[name], meaning=True
                )

            _inputs.append(_tmp)

        return torch.sum(torch.concat(_inputs, dim=-1), dim=-1) + self.bias


class CtrWithAFM(PreTrainedModel):
    """
            模型：AFM
            结构：
            fea1     fea2      fea3     fea4
             |         |         |        |
             H         H         H        H
             |_________|_________|________|
                            |
    sum(A*H1*h2, A*H1*H3, A*H1*h4, A*H2*h3, A*H2*h4, A*H3*h4)
                            |
                           out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.hidden_dim)

        self.logitistic_layer = LogitisticLayer(config.features)

        # define attention
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim, config.attention_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.attention_dim, 1, bias=False),
            torch.nn.Softmax(dim=1),
        )
        self.weight_p = torch.nn.Linear(config.hidden_dim, 1, bias=False)

        # prevent overfitting
        self.dropout1 = torch.nn.Dropout(config.attention_dropout[0])
        self.dropout2 = torch.nn.Dropout(config.attention_dropout[1])

        self.register_buffer(
            "triu_index",
            torch.triu(
                torch.ones(len(config.features), len(config.features)), 1
            ).nonzero(),
        )

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        for layer in self.attention:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        torch.nn.init.xavier_uniform_(self.weight_p.weight)

    def add_regularization(self):
        reg_loss = 0
        for _, param in self.named_parameters():
            reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        _out = self.embedding_layer(**kwargs)

        # stack col fm
        _out = torch.stack(list(_out.values()), dim=1)

        _left = torch.index_select(_out, 1, self.triu_index[:, 0])
        _right = torch.index_select(_out, 1, self.triu_index[:, 1])

        _out = _left * _right

        # attention
        _att_weight = self.attention(_out)
        _att_weight = self.dropout1(_att_weight)
        _out = torch.sum(_att_weight * _out, dim=1)
        _out = self.dropout2(_out)
        _out = self.weight_p(_out)
        _out = torch.sum(_out, dim=1)

        logits = _out + self.logitistic_layer(**kwargs)

        logits = torch.sigmoid(logits)

        labels = None
        if self.label_name in kwargs:
            labels = kwargs[self.label_name]

        if labels is None:
            return CtrModelOutput(logits=logits)

        shift_labels = labels.float().contiguous().view(-1)
        shift_logits = logits.contiguous().view(-1)
        loss = self.criterion(shift_logits, shift_labels) + self.add_regularization()

        return CtrModelOutput(
            loss=loss,
            logits=shift_logits,
        )
