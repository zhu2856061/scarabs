# -*- coding: utf-8 -*-
# @Time   : 2024/07/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class CtrModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CtrWithFiBiNETConfig(PretrainedConfig):
    model_type = "CtrWithNFM"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_size=8,
        mlp_hidden_units=[64, 64, 64],
        reduction_ratio=3,
        bilinear_type="field_interaction",
        dropout_rates=0,
        batch_norm=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_size = embedding_size
        self.mlp_hidden_units = mlp_hidden_units
        self.reduction_ratio = reduction_ratio
        self.bilinear_type = bilinear_type
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


class LogitisticLayer(torch.nn.Module):
    def __init__(self, features) -> None:
        super().__init__()

        self.features = features

        self.feature_w = torch.nn.ModuleDict()
        for name, feature in self.features.items():
            if feature["shared_embed_name"] is not None:
                continue
            self.feature_w[name] = EmbeddingUnitLayer(len(feature["vocab"]), 1)

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

        return torch.sum(torch.concat(_inputs, dim=-1), dim=-1)


class SENetLayer(torch.nn.Module):
    def __init__(self, num_fields, reduction_ratio=3):
        super(SENetLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(num_fields, reduced_size, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(reduced_size, num_fields, bias=False),
            torch.nn.ReLU(),
        )
        self._init_weight_()

    def _init_weight_(self):
        for layer in self.excitation:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class BilinearInteractionLayer(torch.nn.Module):
    def __init__(self, num_fields, embedding_dim, bilinear_type="field_interaction"):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = torch.nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = torch.nn.ModuleList(
                [
                    torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
                    for i in range(num_fields)
                ]
            )
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = torch.nn.ModuleList(
                [
                    torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
                    for i, j in combinations(range(num_fields), 2)
                ]
            )
        else:
            raise NotImplementedError()

    def _init_weight_(self):
        if self.bilinear_type == "field_all":
            torch.nn.init.xavier_uniform_(self.bilinear_layer.weight)

        elif isinstance(self.bilinear_layer, torch.nn.ModuleList):
            for layer in self.bilinear_layer:
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [
                self.bilinear_layer(v_i) * v_j
                for v_i, v_j in combinations(feature_emb_list, 2)
            ]
        elif self.bilinear_type == "field_each" and isinstance(
            self.bilinear_layer, torch.nn.ModuleList
        ):
            bilinear_list = [
                self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                for i, j in combinations(range(len(feature_emb_list)), 2)
            ]
        elif self.bilinear_type == "field_interaction" and isinstance(
            self.bilinear_layer, torch.nn.ModuleList
        ):
            bilinear_list = [
                self.bilinear_layer[i](v[0]) * v[1]
                for i, v in enumerate(combinations(feature_emb_list, 2))
            ]
        return torch.cat(bilinear_list, dim=1)


class CtrWithFiBiNET(PreTrainedModel):
    """
    模型：FiBiNET
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
         |                  |
       senet          bilinear_interaction
         |                     |
    bilinear_interaction       |
         |_____________________|
                  |
                 mlp
                  |
                 out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.mlp_hidden_units = config.mlp_hidden_units

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_size)

        self.senet_layer = SENetLayer(len(config.features), config.reduction_ratio)

        self.bilinear_layer = BilinearInteractionLayer(
            len(config.features), config.embedding_size, config.bilinear_type
        )

        self.mlp_layer = MlpLayer(
            config.embedding_size * len(config.features) * (len(config.features) - 1),
            config.mlp_hidden_units,
            config.batch_norm,
            config.dropout_rates,
        )

        self.logitistic_layer = LogitisticLayer(config.features)

        self.out = torch.nn.Linear(config.mlp_hidden_units[-1], 1)
        # loss
        self.criterion = torch.nn.BCELoss()

    def add_regularization(self):
        reg_loss = 0
        for _, param in self.named_parameters():
            reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        _out = self.embedding_layer(**kwargs)
        _out = torch.stack(list(_out.values()), dim=1)

        # senet
        _senet_out = self.senet_layer(_out)

        # bilinear
        bilinear_p = self.bilinear_layer(_out)
        bilinear_q = self.bilinear_layer(_senet_out)

        _out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)

        _out = self.mlp_layer(_out)
        _out = self.out(_out)
        _out = torch.sum(_out, dim=1)

        logits = _out + self.logitistic_layer(**kwargs)  # bias added in self.mlp_layer

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
