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


class CtrWithEDCNConfig(PretrainedConfig):
    model_type = "CtrWithEDCN"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        num_cross_layers=8,
        bridge_type="hadamard_product",
        use_regulation_module=False,
        temperature=1,
        dropout_rates=0,
        batch_norm=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.num_cross_layers = num_cross_layers
        self.bridge_type = bridge_type
        self.use_regulation_module = use_regulation_module
        self.temperature = temperature
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

        return torch.sum(torch.concat(_inputs, dim=-1), dim=-1) + self.bias


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


class BridgeModule(torch.nn.Module):
    def __init__(self, hidden_dim, bridge_type="hadamard_product"):
        super(BridgeModule, self).__init__()
        assert bridge_type in [
            "hadamard_product",
            "pointwise_addition",
            "concatenation",
            "attention_pooling",
        ], "bridge_type={} is not supported.".format(bridge_type)
        self.bridge_type = bridge_type
        if bridge_type == "concatenation":
            self.concat_pooling = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim), torch.nn.ReLU()
            )
        elif bridge_type == "attention_pooling":
            self.attention1 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
                torch.nn.Softmax(dim=-1),
            )
            self.attention2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
                torch.nn.Softmax(dim=-1),
            )

    def _init_weight_(self):
        if self.bridge_type == "concatenation":
            for layer in self.concat_pooling:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
        elif self.bridge_type == "attention_pooling":
            for layer in self.attention1:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

            for layer in self.attention2:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, X1, X2):
        out = None
        if self.bridge_type == "hadamard_product":
            out = X1 * X2
        elif self.bridge_type == "pointwise_addition":
            out = X1 + X2
        elif self.bridge_type == "concatenation":
            out = self.concat_pooling(torch.cat([X1, X2], dim=-1))
        elif self.bridge_type == "attention_pooling":
            out = self.attention1(X1) * X1 + self.attention1(X2) * X2
        return out


class RegulationModule(torch.nn.Module):
    def __init__(
        self, num_fields, embedding_dim, tau=1, use_bn=False, use_regulation=True
    ):
        super(RegulationModule, self).__init__()
        self.use_regulation = use_regulation
        self.use_bn = use_bn
        if self.use_regulation:
            self.tau = tau
            self.embedding_dim = embedding_dim
            self.g1 = torch.nn.Parameter(torch.ones(num_fields))
            self.g2 = torch.nn.Parameter(torch.ones(num_fields))
        if self.use_bn:
            self.bn1 = torch.nn.BatchNorm1d(num_fields * embedding_dim)
            self.bn2 = torch.nn.BatchNorm1d(num_fields * embedding_dim)

    def forward(self, X):
        if self.use_regulation:
            g1 = (
                (self.g1 / self.tau)
                .softmax(dim=-1)
                .unsqueeze(-1)
                .repeat(1, self.embedding_dim)
                .view(1, -1)
            )
            g2 = (
                (self.g2 / self.tau)
                .softmax(dim=-1)
                .unsqueeze(-1)
                .repeat(1, self.embedding_dim)
                .view(1, -1)
            )
            out1, out2 = g1 * X, g2 * X
        else:
            out1, out2 = X, X
        if self.use_bn:
            out1, out2 = self.bn1(out1), self.bn2(out2)
        return out1, out2


class CrossInteractionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = torch.nn.Linear(input_dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))

        self._init_weight_()

    def _init_weight_(self):
        """Fine tune details: init weight very important !!!"""
        torch.nn.init.xavier_normal_(self.weight.weight)

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


class CtrWithEDCN(PreTrainedModel):
    """
    模型：AFM
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
                    |
                Regulation Module
                    |
                cross interaction
                    |
                Bridge Module
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

        self.dense_layers = torch.nn.ModuleList(
            [
                MlpLayer(
                    self.hidden_dim,
                    [self.hidden_dim],
                    False,
                    config.dropout_rates,
                )
                for _ in range(config.num_cross_layers)
            ]
        )
        self.cross_layers = torch.nn.ModuleList(
            [
                CrossInteractionLayer(self.hidden_dim)
                for _ in range(config.num_cross_layers)
            ]
        )
        self.bridge_modules = torch.nn.ModuleList(
            [
                BridgeModule(self.hidden_dim, config.bridge_type)
                for _ in range(config.num_cross_layers)
            ]
        )
        self.regulation_modules = torch.nn.ModuleList(
            [
                RegulationModule(
                    self.num_fields,
                    config.embedding_dim,
                    tau=config.temperature,
                    use_bn=config.batch_norm,
                    use_regulation=config.use_regulation_module,
                )
                for _ in range(config.num_cross_layers)
            ]
        )

        self.out = torch.nn.Linear(self.hidden_dim * 3, 1)

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def add_regularization(self):
        reg_loss = 0
        for _, param in self.named_parameters():
            reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        _out = self.embedding_layer(**kwargs)

        # stack col fm
        _out = torch.stack(list(_out.values()), dim=1)

        cross_i, deep_i = self.regulation_modules[0](_out.flatten(start_dim=1))

        cross_0 = cross_i
        bridge_i = None
        for i in range(len(self.cross_layers)):
            if i > 0:
                cross_i, deep_i = self.regulation_modules[i](bridge_i)
            cross_i = cross_i + self.cross_layers[i](cross_0, cross_i)
            deep_i = self.dense_layers[i](deep_i)
            bridge_i = self.bridge_modules[i](cross_i, deep_i)

        logits = self.out(torch.cat([cross_i, deep_i, bridge_i], dim=-1))  # type: ignore

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
