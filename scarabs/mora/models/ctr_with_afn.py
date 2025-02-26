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


class CtrWithAFNConfig(PretrainedConfig):
    model_type = "CtrWithAFN"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        ensemble_dnn=True,
        dnn_hidden_units=[64, 64, 64],
        afn_hidden_units=[64, 64, 64],
        logarithmic_neurons=5,
        batch_norm=False,
        dropout_rates=0.1,
        regularizer=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.ensemble_dnn = ensemble_dnn
        self.dnn_hidden_units = dnn_hidden_units
        self.afn_hidden_units = afn_hidden_units
        self.logarithmic_neurons = logarithmic_neurons

        self.batch_norm = batch_norm
        self.dropout_rates = dropout_rates
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


class CtrWithAFN(PreTrainedModel):
    """
    模型：afn
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
          |                 |
      logarithmic_net      mlp
          |                 |
         mlp                |
          |_________________|
                    |
                   out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        if config.features is None:
            raise ValueError("config features must be provided")
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.dnn_hidden_units = config.dnn_hidden_units
        self.num_fields = len(config.features)

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)

        self.coefficient_W = torch.nn.Linear(
            self.num_fields, config.logarithmic_neurons, bias=False
        )

        self.dense_layer = MlpLayer(
            config.embedding_dim * config.logarithmic_neurons,
            config.afn_hidden_units,
            config.batch_norm,
            config.dropout_rates,
        )
        final_dim = config.afn_hidden_units[-1]

        self.log_batch_norm = torch.nn.BatchNorm1d(self.num_fields)
        self.exp_batch_norm = torch.nn.BatchNorm1d(config.logarithmic_neurons)
        self.ensemble_dnn = config.ensemble_dnn

        if self.ensemble_dnn:
            self.embedding_layer2 = EmbeddingLayer(
                config.features, config.embedding_dim
            )
            self.dnn = MlpLayer(
                config.embedding_dim * self.num_fields,
                config.dnn_hidden_units,
                config.batch_norm,
                config.dropout_rates,
            )
            final_dim += config.dnn_hidden_units[-1]

        self.out = torch.nn.Linear(final_dim, 1)

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        # torch.nn.init.kaiming_uniform_(self.out.weight, a=1, nonlinearity="sigmoid")
        # torch.nn.init.zeros_(self.out.bias)
        torch.nn.init.xavier_normal_(self.coefficient_W.weight)
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

        _inputs = torch.stack(list(_inputs.values()), dim=1)
        _inputs = self.logarithmic_net(_inputs)
        _afn_out = self.dense_layer(_inputs)

        if self.ensemble_dnn:
            _inputs2 = self.embedding_layer2(**kwargs)
            _inputs2 = torch.stack(list(_inputs2.values()), dim=1)

            _dnn_out = self.dnn(_inputs2.flatten(start_dim=1))

        # out
        _final_out = (
            torch.concat([_afn_out, _dnn_out], dim=-1)
            if self.ensemble_dnn
            else _afn_out
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

    def logarithmic_net(self, feature_emb):
        """
          模型：lnn
          结构：
          x1        x2        x3       x4
           |         |         |        |
        ln(x1)    ln(x2)    ln(x3)   ln(x4)
           |         |         |        |
         norm      norm      norm      norm
           |_________|_________|________|
                          |
                        linear
                          |
                         exp
                          |
                         norm
        """
        feature_emb = torch.abs(feature_emb)
        # because there cannot be negative numbers in logarithms
        feature_emb = torch.clamp(
            feature_emb, min=1e-5
        )  # ReLU with min 1e-5 (better than 1e-7 suggested in paper)
        log_feature_emb = torch.log(feature_emb)  # element-wise log
        log_feature_emb = self.log_batch_norm(
            log_feature_emb
        )  # batch_size * num_fields * embedding_dim
        logarithmic_out = self.coefficient_W(log_feature_emb.transpose(2, 1)).transpose(
            1, 2
        )
        cross_out = torch.exp(logarithmic_out)  # element-wise exp
        cross_out = self.exp_batch_norm(
            cross_out
        )  # batch_size * logarithmic_neurons * embedding_dim
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out
