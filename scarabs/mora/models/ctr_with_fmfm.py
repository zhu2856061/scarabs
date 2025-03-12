# -*- coding: utf-8 -*-
# @Time   : 2024/07/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import torch
from sympy import false
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class CtrModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CtrWithFMFMConfig(PretrainedConfig):
    model_type = "CtrWithFMFM"

    def __init__(
        self,
        features=None,
        label_name="label",
        hidden_dim=8,
        regularizer=5e-06,
        field_interaction_type="matrixed",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.hidden_dim = hidden_dim
        self.regularizer = regularizer
        self.field_interaction_type = field_interaction_type


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


class CtrWithFMFM(PreTrainedModel):
    """
            模型：fmfm
            结构：
            fea1     fea2      fea3     fea4
             |         |         |        |
             H         H         H        H
             |_________|_________|________|
                            |
    sum(H1*Matric*h2, H1*Matric*H3, H1*Matric*h4, H2*Matric*h3, H2*Matric*h4, H3*Matric*h4)
                            |
                           out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.field_interaction_type = config.field_interaction_type

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.hidden_dim)

        #  define lr layer
        self.logitistic_layer = LogitisticLayer(config.features)

        # define matric
        interact_dim = int(len(config.features) * (len(config.features) - 1) / 2)
        if config.field_interaction_type == "vectorized":
            self.interaction_weight = torch.nn.Parameter(
                torch.Tensor(interact_dim, config.hidden_dim)
            ).to(self.logitistic_layer.bias.device)
        else:
            self.interaction_weight = torch.nn.Parameter(
                torch.Tensor(interact_dim, config.hidden_dim, config.hidden_dim)
            ).to(self.logitistic_layer.bias.device)

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
        torch.nn.init.xavier_normal_(self.interaction_weight)

    def add_regularization(self):
        reg_loss = 0
        for _, param in self.named_parameters():
            reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        feature_emb = self.embedding_layer(**kwargs)
        feature_emb = torch.stack(list(feature_emb.values()), dim=1)

        left_emb = torch.index_select(feature_emb, 1, self.triu_index[:, 0])
        right_emb = torch.index_select(feature_emb, 1, self.triu_index[:, 1])

        if self.field_interaction_type == "vectorized":
            left_emb = left_emb * self.interaction_weight
        else:
            left_emb = torch.matmul(
                left_emb.unsqueeze(2), self.interaction_weight
            ).squeeze(2)

        logits = (left_emb * right_emb).sum((1, 2)) + self.logitistic_layer(**kwargs)

        # concat
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
