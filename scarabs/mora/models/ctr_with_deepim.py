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


class CtrWithDeepIMConfig(PretrainedConfig):
    model_type = "CtrWithDeepIM"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        im_order=2,
        im_batch_norm=False,
        dnn_hidden_units=[64, 64, 64],
        dropout_rates=0,
        batch_norm=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.im_order = im_order
        self.im_batch_norm = im_batch_norm
        self.dnn_hidden_units = dnn_hidden_units
        self.dropout_rates = dropout_rates
        self.batch_norm = batch_norm
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


class MlpUnitLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm, dropout_rates):
        super(MlpUnitLayer, self).__init__()

        self.batch_norm = batch_norm
        self.dropout_rates = dropout_rates

        self._linear = torch.nn.Linear(input_dim, output_dim)

        if self.batch_norm:
            self._bn = torch.nn.BatchNorm1d(output_dim)

        self._relu = torch.nn.ReLU()

        if self.dropout_rates > 0:
            self._dropout = torch.nn.Dropout(self.dropout_rates)

        self._init_weight_()

    def _init_weight_(self):
        """Fine tune details: init weight very important !!!"""
        torch.nn.init.xavier_normal_(self._linear.weight)
        torch.nn.init.zeros_(self._linear.bias)

    def forward(self, X):
        _out = X

        _out = self._linear(_out)

        if self.batch_norm:
            _out = self._bn(_out)

        _out = self._relu(_out)

        if self.dropout_rates > 0:
            _out = self._dropout(_out)

        return _out


class MlpLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_units, batch_norm, dropout_rates):
        super(MlpLayer, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_units)):
            if i == 0:
                self.hidden_layers.append(
                    MlpUnitLayer(input_dim, hidden_units[i], batch_norm, dropout_rates)
                )
            else:
                self.hidden_layers.append(
                    MlpUnitLayer(
                        hidden_units[i - 1], hidden_units[i], batch_norm, dropout_rates
                    )
                )

    def forward(self, X):
        for i in range(len(self.hidden_layers)):
            X = self.hidden_layers[i](X)

        return X


class InteractionMachine(torch.nn.Module):
    def __init__(self, embedding_dim, order=2, batch_norm=False):
        super(InteractionMachine, self).__init__()
        assert order < 6, "order={} is not supported.".format(order)
        self.order = order
        self.bn = torch.nn.BatchNorm1d(embedding_dim * order) if batch_norm else None
        self.fc = torch.nn.Linear(order * embedding_dim, 1)
        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def second_order(self, p1, p2):
        return (p1.pow(2) - p2) / 2

    def third_order(self, p1, p2, p3):
        return (p1.pow(3) - 3 * p1 * p2 + 2 * p3) / 6

    def fourth_order(self, p1, p2, p3, p4):
        return (
            p1.pow(4) - 6 * p1.pow(2) * p2 + 3 * p2.pow(2) + 8 * p1 * p3 - 6 * p4
        ) / 24

    def fifth_order(self, p1, p2, p3, p4, p5):
        return (
            p1.pow(5)
            - 10 * p1.pow(3) * p2
            + 20 * p1.pow(2) * p3
            - 30 * p1 * p4
            - 20 * p2 * p3
            + 15 * p1 * p2.pow(2)
            + 24 * p5
        ) / 120

    def forward(self, X):
        out = []
        Q = X
        if self.order >= 1:
            p1 = Q.sum(dim=1)
            out.append(p1)
            if self.order >= 2:
                Q = Q * X
                p2 = Q.sum(dim=1)
                out.append(self.second_order(p1, p2))
                if self.order >= 3:
                    Q = Q * X
                    p3 = Q.sum(dim=1)
                    out.append(self.third_order(p1, p2, p3))
                    if self.order >= 4:
                        Q = Q * X
                        p4 = Q.sum(dim=1)
                        out.append(self.fourth_order(p1, p2, p3, p4))
                        if self.order == 5:
                            Q = Q * X
                            p5 = Q.sum(dim=1)
                            out.append(self.fifth_order(p1, p2, p3, p4, p5))
        out = torch.cat(out, dim=-1)
        if self.bn is not None:
            out = self.bn(out)
        y = self.fc(out)
        return y


class CtrWithDeepIM(PreTrainedModel):
    """
    模型：DeepFM
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
     |                        |
     order                   DNN
     |________________________|
                   |
                   out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.num_fields = len(config.features)
        self.hidden_dim = self.num_fields * config.embedding_dim

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)

        self.im_layer = InteractionMachine(
            config.embedding_dim, config.im_order, config.im_batch_norm
        )

        self.mlp_layer = MlpLayer(
            self.hidden_dim,
            config.dnn_hidden_units,
            config.batch_norm,
            config.dropout_rates,
        )

        self.out = torch.nn.Linear(config.dnn_hidden_units[-1], 1)

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def add_regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "embedding" in name or "Embedding" in name:
                reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        _out = self.embedding_layer(**kwargs)

        # stack col fm
        _out = torch.stack(list(_out.values()), dim=1)

        _im_out = self.im_layer(_out)

        _dnn_out = self.mlp_layer(_out.flatten(start_dim=1))
        _dnn_out = self.out(_dnn_out)

        logits = torch.sigmoid(_im_out + _dnn_out)

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
