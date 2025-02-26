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


class CtrWithFinalNetConfig(PretrainedConfig):
    model_type = "CtrWithFinalNet"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        block_type="2B",
        use_field_gate=True,
        block1_hidden_units=[64, 64, 64],
        block1_dropout=0,
        block2_hidden_units=[64, 64, 64],
        block2_dropout=0,
        residual_type="concat",
        batch_norm=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.block_type = block_type
        self.use_field_gate = use_field_gate
        self.block1_hidden_units = block1_hidden_units
        self.block1_dropout = block1_dropout
        self.block2_hidden_units = block2_hidden_units
        self.block2_dropout = block2_dropout
        self.residual_type = residual_type

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


class FieldGate(torch.nn.Module):
    def __init__(self, num_fields):
        super(FieldGate, self).__init__()
        self.proj_field = torch.nn.Linear(num_fields, num_fields)
        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_uniform_(self.proj_field.weight)
        torch.nn.init.zeros_(self.proj_field.bias)

    def forward(self, feature_emb):
        gates = self.proj_field(feature_emb.transpose(1, 2)).transpose(1, 2)
        out = torch.cat([feature_emb, feature_emb * gates], dim=1)  # b x 2f x d
        return out


class FinalBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=[],
        dropout_rates=0,
        batch_norm=True,
        residual_type="sum",
    ):
        # Replacement of MLP_Block, identical when order=1
        super(FinalBlock, self).__init__()

        self.layer = torch.nn.ModuleList()
        self.norm = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        self.activation = torch.nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.layer.append(
                FactorizedInteraction(
                    hidden_units[idx],
                    hidden_units[idx + 1],
                    residual_type=residual_type,
                )
            )
            if batch_norm:
                self.norm.append(torch.nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates > 0:
                self.dropout.append(torch.nn.Dropout(dropout_rates))
            self.activation.append(torch.nn.ReLU())

    def forward(self, X):
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return X_i


class FactorizedInteraction(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, bias=True, residual_type="sum", activation=None
    ):
        """A replacement of nn.Linear to enhance multiplicative feature interactions.
        `residual_type="concat"` uses the same number of parameters as nn.Linear
        while `residual_type="sum"` doubles the number of parameters.
        """
        super(FactorizedInteraction, self).__init__()
        self.residual_type = residual_type
        if residual_type == "sum":
            output_dim = output_dim * 2
        else:
            assert output_dim % 2 == 0, "output_dim should be divisible by 2."
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)
        self.activation = torch.nn.ReLU()

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        if self.activation is not None:
            h1 = self.activation(h1)
        if self.residual_type == "concat":
            h = torch.cat([h2, h1 * h2], dim=-1)
        elif self.residual_type == "sum":
            h = h2 + h1 * h2
        return h


class CtrWithFinalNet(PreTrainedModel):
    """
    模型：FinalNet
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
     |                            |
     field gate                  field gate
     |                            |
     final block A              final block B
     |____________________________|
     |                            |
     y1      y = 1/2(y1+y2)       y2
     |            |               |
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        assert config.block_type in ["1B", "2B"], "block_type={} not supported.".format(
            config.block_type
        )

        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.num_fields = len(config.features)
        self.hidden_dim = self.num_fields * config.embedding_dim
        self.use_field_gate = config.use_field_gate

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)

        if self.use_field_gate:
            self.field_gate = FieldGate(self.num_fields)
            gate_out_dim = config.embedding_dim * self.num_fields * 2

        self.block_type = config.block_type
        self.block1 = FinalBlock(
            input_dim=gate_out_dim
            if config.use_field_gate
            else config.embedding_dim * self.num_fields,
            hidden_units=config.block1_hidden_units,
            dropout_rates=config.block1_dropout,
            batch_norm=config.batch_norm,
            residual_type=config.residual_type,
        )

        self.fc1 = torch.nn.Linear(config.block1_hidden_units[-1], 1)

        if config.block_type == "2B":
            self.block2 = FinalBlock(
                input_dim=config.embedding_dim * self.num_fields,
                hidden_units=config.block2_hidden_units,
                dropout_rates=config.block2_dropout,
                batch_norm=config.batch_norm,
                residual_type=config.residual_type,
            )
            self.fc2 = torch.nn.Linear(config.block2_hidden_units[-1], 1)

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        if self.block_type == "2B":
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc2.bias)

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

        if self.block_type == "1B":
            if self.use_field_gate:
                _out = self.field_gate(_out)
            block1_out = self.block1(_out.flatten(start_dim=1))
            logits = self.fc1(block1_out)

        elif self.block_type == "2B":
            if self.use_field_gate:
                _out_1 = self.field_gate(_out)
            block1_out = self.block1(_out_1.flatten(start_dim=1))
            y1 = self.fc1(block1_out)

            block2_out = self.block2(_out.flatten(start_dim=1))
            y2 = self.fc2(block2_out)

            logits = 0.5 * (y1 + y2)

        logits = torch.sigmoid(logits)

        labels = None
        if self.label_name in kwargs:
            labels = kwargs[self.label_name]

        if labels is None:
            return CtrModelOutput(logits=logits)

        shift_labels = labels.float().contiguous().view(-1)
        shift_logits = logits.contiguous().view(-1)
        loss = self.criterion(shift_logits, shift_labels) + self.add_regularization()
        if self.block_type == "2B":
            y1 = torch.sigmoid(y1)
            loss1 = (
                self.criterion(y1.contiguous().view(-1), shift_labels)
                + self.add_regularization()
            )
            y2 = torch.sigmoid(y2)
            loss2 = (
                self.criterion(y2.contiguous().view(-1), shift_labels)
                + self.add_regularization()
            )
            loss = loss + loss1 + loss2
        return CtrModelOutput(
            loss=loss,
            logits=shift_logits,
        )
