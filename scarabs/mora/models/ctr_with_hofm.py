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


class CtrWithHOFMConfig(PretrainedConfig):
    model_type = "CtrWithHOFM"

    def __init__(
        self,
        features=None,
        label_name="label",
        hidden_dim=8,
        order=3,
        reuse_embedding=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.hidden_dim = hidden_dim
        self.regularizer = regularizer
        self.order = order
        self.reuse_embedding = reuse_embedding
        assert self.order > 2


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


class CtrWithHOFM(PreTrainedModel):
    """
        模型：lr
        结构：
        fea1     fea2      fea3     fea4
         |         |         |        |
         H         H         H        H
         |_________|_________|________|
                        |
    sum(H1*h2, H1*H3, H1*h4, H2*h3, H2*h4, H3*h4)
    + sum(H1*h2*H3, H1*H2*H4, H1*H3*H4, H2*H3*H4)
                        |
                       out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.order = config.order
        self.reuse_embedding = config.reuse_embedding

        #  define embedding layer
        if self.reuse_embedding:
            self.embedding_layer = torch.nn.ModuleList(
                [EmbeddingLayer(config.features, config.hidden_dim)]
            )
        else:
            self.embedding_layer = torch.nn.ModuleList(
                [
                    EmbeddingLayer(config.features, config.hidden_dim)
                    for _ in range(self.order - 1)
                ]
            )

        #  define lr layer
        self.logitistic_layer = LogitisticLayer(config.features)

        # get device

        # order
        # self.field_conjunction_dict = dict()
        # for order_i in range(2, self.order + 1):
        #     order_i_conjunction = zip(
        #         *list(combinations(range(len(config.features)), order_i))
        #     )
        #     for k, field_index in enumerate(order_i_conjunction):
        #         self.field_conjunction_dict[(order_i, k)] = torch.LongTensor(
        #             field_index
        #         ).to(device)

        # loss
        self.criterion = torch.nn.BCELoss()

    def add_regularization(self):
        reg_loss = 0
        for _, param in self.named_parameters():
            reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def high_order_interaction(self, feature_emb, order_i):
        _out = torch.stack(list(feature_emb.values()), dim=1)

        order_i_conjunction = list(
            zip(*list(combinations(range(len(feature_emb)), order_i)))
        )
        device = _out.device
        hadamard_product = torch.index_select(
            _out, 1, torch.LongTensor(order_i_conjunction[0]).to(device)
        )

        for k in range(1, order_i):
            hadamard_product = hadamard_product * torch.index_select(
                _out, 1, torch.LongTensor(order_i_conjunction[k]).to(device)
            )

        hadamard_product = hadamard_product.sum((1, 2))

        return hadamard_product

    def forward(self, **kwargs):
        logits = self.logitistic_layer(**kwargs)

        if self.reuse_embedding:
            _emb_out = self.embedding_layer[0](**kwargs)
            for i in range(2, self.order + 1):
                order_i_out = self.high_order_interaction(
                    _emb_out,
                    order_i=i,
                )
                logits += order_i_out
        else:
            for i in range(2, self.order + 1):
                _emb_out = self.embedding_layer[i - 2](**kwargs)
                order_i_out = self.high_order_interaction(
                    _emb_out,
                    order_i=i,
                )
                logits += order_i_out

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
