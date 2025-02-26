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


class CtrWithDCNConfig(PretrainedConfig):
    model_type = "CtrWithDCN"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        num_cross_layers=3,
        batch_norm=False,
        dropout_rates=0.1,
        dnn_hidden_units=[],
        regularizer=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.num_cross_layers = num_cross_layers
        self.batch_norm = batch_norm
        self.dropout_rates = dropout_rates
        self.dnn_hidden_units = dnn_hidden_units
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


class CrossNet(torch.nn.Module):
    """
    模型: cross_net:
    结构:
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
          |                 |
        X_0 * X_i * W + b + X_i
          |_________________|
                    |
                   out

    """

    def __init__(self, input_dim):
        super(CrossNet, self).__init__()
        self._weight = torch.nn.Linear(input_dim, 1, bias=False)
        self._bias = torch.nn.Parameter(torch.zeros(input_dim))

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self._weight.weight)

    def forward(self, X_0, X_i):
        _out = self._weight(X_i) * X_0 + self._bias + X_i
        return _out


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


class CtrWithDCN(PreTrainedModel):
    """
    模型：dcn
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
          |                 |
      cross net            mlp
          |_________________|
                    |
                   out


    """

    def __init__(self, config: CtrWithDCNConfig):
        super().__init__(config)
        if config.features is None:
            raise ValueError("config features must be provided")
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.dnn_hidden_units = config.dnn_hidden_units

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)
        input_dim = len(config.features) * config.embedding_dim

        # define cross layer
        self.cross_layer = torch.nn.ModuleDict()
        for i in range(config.num_cross_layers):
            self.cross_layer[f"cross_{i}"] = CrossNet(input_dim)

        # define mlp layer
        if len(self.dnn_hidden_units) > 0:
            self.mlp_layer = MlpLayer(
                input_dim,
                config.dnn_hidden_units,
                config.batch_norm,
                config.dropout_rates,
            )

        # out
        final_dim = input_dim
        if len(self.dnn_hidden_units) > 0:
            final_dim += self.dnn_hidden_units[-1]
        self.out = torch.nn.Linear(final_dim, 1)

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        # torch.nn.init.kaiming_uniform_(self.out.weight, a=1, nonlinearity="sigmoid")
        # torch.nn.init.zeros_(self.out.bias)
        torch.nn.init.xavier_normal_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def add_regularization(self):
        """Fine tune details: regular very important !!!"""
        reg_loss = 0
        for name, param in self.named_parameters():
            if "embedding" in name or "Embedding" in name:
                reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        _inputs = self.embedding_layer(**kwargs)

        _inputs = torch.concat(list(_inputs.values()), dim=1)

        # cross net
        X_0 = _inputs
        _cross_out = X_0
        for i in range(len(self.cross_layer)):
            _cross_out = self.cross_layer[f"cross_{i}"](X_0, _cross_out)

        # deep net
        if len(self.dnn_hidden_units) > 0:
            _deep_out = self.mlp_layer(_inputs)

        # out
        _final_out = (
            _cross_out
            if len(self.dnn_hidden_units) <= 0
            else torch.concat([_cross_out, _deep_out], dim=1)
        )
        logits = self.out(_final_out)
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
