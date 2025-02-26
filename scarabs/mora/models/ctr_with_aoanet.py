# -*- coding: utf-8 -*-
# @Time   : 2024/07/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class CtrModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CtrWithAOANetConfig(PretrainedConfig):
    model_type = "CtrWithAOANet"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        dnn_hidden_units=[64, 64, 64],
        num_interaction_layers=3,
        num_subspaces=4,
        dropout_rates=0,
        batch_norm=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.num_interaction_layers = num_interaction_layers
        self.num_subspaces = num_subspaces

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

        return (
            torch.sum(torch.concat(_inputs, dim=-1), dim=-1, keepdim=True) + self.bias
        )


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


class GeneralizedInteractionNet(torch.nn.Module):
    def __init__(self, num_layers, num_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteractionNet, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                GeneralizedInteraction(
                    num_fields if i == 0 else num_subspaces,
                    num_subspaces,
                    num_fields,
                    embedding_dim,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, B_0):
        B_i = B_0
        for layer in self.layers:
            B_i = layer(B_0, B_i)
        return B_i


class GeneralizedInteraction(torch.nn.Module):
    def __init__(self, input_subspaces, output_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteraction, self).__init__()
        self.input_subspaces = input_subspaces
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.W = torch.nn.Parameter(
            torch.eye(embedding_dim, embedding_dim)
            .unsqueeze(0)
            .repeat(output_subspaces, 1, 1)
        )
        self.alpha = torch.nn.Parameter(
            torch.ones(input_subspaces * num_fields, output_subspaces)
        )
        self.h = torch.nn.Parameter(torch.ones(output_subspaces, embedding_dim, 1))

    def forward(self, B_0, B_i):
        outer_product = torch.einsum(
            "bnh,bnd->bnhd",
            B_0.repeat(1, self.input_subspaces, 1),
            B_i.repeat(1, 1, self.num_fields).view(B_i.size(0), -1, self.embedding_dim),
        )  # b x (field*in) x d x d
        fusion = torch.matmul(
            outer_product.permute(0, 2, 3, 1), self.alpha
        )  # b x d x d x out
        fusion = self.W * fusion.permute(0, 3, 1, 2)  # b x out x d x d
        B_i = torch.matmul(fusion, self.h).squeeze(-1)  # b x out x d
        return B_i


class CtrWithAOANet(PreTrainedModel):
    """
    模型：AOANet
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
     |                            |
     gin                        DNN
     |____________________________|
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

        self.dnn_layer = MlpLayer(
            self.hidden_dim,
            config.dnn_hidden_units,
            config.batch_norm,
            config.dropout_rates,
        )

        self.gin_layer = GeneralizedInteractionNet(
            config.num_interaction_layers,
            config.num_subspaces,
            self.num_fields,
            config.embedding_dim,
        )

        self.out = torch.nn.Linear(
            config.dnn_hidden_units[-1] + config.num_subspaces * config.embedding_dim, 1
        )

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

        dnn_out = self.dnn_layer(_out.flatten(start_dim=1))
        interact_out = self.gin_layer(_out).flatten(start_dim=1)

        logits = self.out(torch.cat([dnn_out, interact_out], dim=-1))

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
