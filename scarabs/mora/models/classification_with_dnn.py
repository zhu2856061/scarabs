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
class ClassificationModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class ClassificationWithDNNConfig(PretrainedConfig):
    model_type = "ClassificationWithDNN"

    def __init__(
        self,
        features=None,
        embedding_size=64,
        feature_hidden_units=[64, 64],
        mlp_hidden_units=[64, 64],
        regularizer=5e-06,
        batch_norm=False,
        dropout_rates=0.0,
        label_name="label",
        num_labels=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.embedding_size = embedding_size
        self.feature_hidden_units = feature_hidden_units
        self.mlp_hidden_units = mlp_hidden_units
        self.regularizer = regularizer
        self.batch_norm = batch_norm
        self.dropout_rates = dropout_rates
        self.label_name = label_name
        self.num_labels = num_labels


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


class ClassificationWithDNN(PreTrainedModel):
    """
    模型：feature-mutli-tower-mlp
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
    mlp       mlp       mlp      mlp
     |_________|_________|________|
                    |
                  concat
                    |
                   mlp
                    |
                   out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        #
        self.regularizer = config.regularizer
        self.label_name = config.label_name
        self.num_labels = config.num_labels

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_size)

        # define feature mlp layer
        self.feature_mlp_layers = torch.nn.ModuleDict()
        for name, _ in config.features.items():
            self.feature_mlp_layers[name] = MlpLayer(
                config.embedding_size,
                config.feature_hidden_units,
                config.batch_norm,
                config.dropout_rates,
            )

        # define mlp layer
        self.mlp_layer = MlpLayer(
            config.feature_hidden_units[-1] * len(config.features),
            config.mlp_hidden_units,
            config.batch_norm,
            config.dropout_rates,
        )

        # out
        self.out = torch.nn.Linear(config.mlp_hidden_units[-1], self.num_labels)

        # loss
        self.criterion = torch.nn.CrossEntropyLoss()

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.kaiming_uniform_(self.out.weight, a=1, nonlinearity="sigmoid")
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, **kwargs):
        #
        _inputs = self.embedding_layer(**kwargs)

        #
        _out = []
        for k, v in _inputs.items():
            _out.append(self.feature_mlp_layers[k](v))

        #
        logits = self.mlp_layer(torch.concat(_out, dim=-1))

        logits = self.out(logits)
        logits = torch.sigmoid(logits)
        logits = logits.contiguous().view(-1, self.num_labels)

        labels = None
        if self.label_name in kwargs:
            labels = kwargs[self.label_name]

        if labels is None:
            return ClassificationModelOutput(logits=logits)

        loss = self.criterion(logits, labels)

        return ClassificationModelOutput(
            loss=loss,
            logits=logits,
        )
