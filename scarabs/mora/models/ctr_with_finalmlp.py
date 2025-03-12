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


class CtrWithFinalMLPConfig(PretrainedConfig):
    model_type = "CtrWithFinalMLP"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        mlp1_hidden_units=[64, 64, 64],
        mlp2_hidden_units=[64, 64, 64],
        use_fs=True,
        fs_hidden_units=[64],
        fs1_context=[],
        fs2_context=[],
        num_heads=1,
        dropout_rates=0,
        batch_norm=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.mlp1_hidden_units = mlp1_hidden_units
        self.mlp2_hidden_units = mlp2_hidden_units
        self.use_fs = use_fs
        self.fs_hidden_units = fs_hidden_units
        self.fs1_context = fs1_context
        self.fs2_context = fs2_context
        self.num_heads = num_heads

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
    def __init__(
        self, input_dim, output_dim, batch_norm, user_activation, dropout_rates
    ):
        super(MlpUnitLayer, self).__init__()

        self.batch_norm = batch_norm
        self.user_activation = user_activation
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

        if self.user_activation:
            _out = self._relu(_out)

        if self.dropout_rates > 0:
            _out = self._dropout(_out)

        return _out


class MlpLayer(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_units, batch_norm, user_activation, dropout_rates
    ):
        super(MlpLayer, self).__init__()

        layers = [input_dim] + hidden_units
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            if i == len(layers) - 2:
                self.hidden_layers.append(
                    MlpUnitLayer(
                        layers[i],
                        layers[i + 1],
                        batch_norm,
                        user_activation,
                        dropout_rates,
                    )
                )
            else:
                self.hidden_layers.append(
                    MlpUnitLayer(
                        layers[i],
                        layers[i + 1],
                        batch_norm,
                        True,
                        dropout_rates,
                    )
                )

    def forward(self, X):
        for i in range(len(self.hidden_layers)):
            X = self.hidden_layers[i](X)

        return X


class FeatureSelection(torch.nn.Module):
    def __init__(
        self,
        features,
        feature_dim,
        embedding_dim,
        fs_hidden_units=[],
        fs1_context=[],
        fs2_context=[],
    ):
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        if len(fs1_context) == 0:
            self.fs1_ctx_bias = torch.nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            fs1_context_features = {
                k: v for k, v in features.items() if k in fs1_context
            }

            self.fs1_ctx_emb = EmbeddingLayer(fs1_context_features, embedding_dim)

        self.fs2_context = fs2_context
        if len(fs2_context) == 0:
            self.fs2_ctx_bias = torch.nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            fs2_context_features = {
                k: v for k, v in features.items() if k in fs2_context
            }
            self.fs2_ctx_emb = EmbeddingLayer(fs2_context_features, embedding_dim)

        self.fs1_gate = MlpLayer(
            embedding_dim * max(1, len(fs1_context)),
            fs_hidden_units,
            False,
            True,
            0,
        )
        self.fs1_out = torch.nn.Linear(fs_hidden_units[-1], feature_dim)

        self.fs2_gate = MlpLayer(
            embedding_dim * max(1, len(fs2_context)),
            fs_hidden_units,
            False,
            True,
            0,
        )
        self.fs2_out = torch.nn.Linear(fs_hidden_units[-1], feature_dim)

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self.fs1_out.weight)
        torch.nn.init.xavier_normal_(self.fs2_out.weight)

    def forward(self, X, flat_emb):
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        gt1 = torch.sigmoid(self.fs1_out(self.fs1_gate(fs1_input))) * 2
        feature1 = flat_emb * gt1

        if len(self.fs2_context) == 0:
            fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
        gt2 = torch.sigmoid(self.fs2_out(self.fs2_gate(fs2_input))) * 2
        feature2 = flat_emb * gt2

        return feature1, feature2


class InteractionAggregation(torch.nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert (
            x_dim % num_heads == 0 and y_dim % num_heads == 0
        ), "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = torch.nn.Linear(x_dim, output_dim)
        self.w_y = torch.nn.Linear(y_dim, output_dim)
        self.w_xy = torch.nn.Parameter(
            torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, output_dim)
        )

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self.w_xy)

        torch.nn.init.xavier_normal_(self.w_x.weight)
        torch.nn.init.xavier_normal_(self.w_y.weight)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(
            torch.matmul(
                head_x.unsqueeze(2), self.w_xy.view(self.num_heads, self.head_x_dim, -1)
            ).view(-1, self.num_heads, self.output_dim, self.head_y_dim),
            head_y.unsqueeze(-1),
        ).squeeze(-1)
        output += xy.sum(dim=1)
        return output


class CtrWithFinalMLP(PreTrainedModel):
    """
    模型：FinalMLP
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
                  |
               fs_module
     _____________|________________
     |                            |
     mlp                         mlp
     |____________________________|
                   |
            fusion_module
                   |
                   out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.num_fields = len(config.features)
        self.hidden_dim = self.num_fields * config.embedding_dim
        self.use_fs = config.use_fs

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)

        self.mlp1 = MlpLayer(
            self.hidden_dim,
            config.mlp1_hidden_units,
            config.batch_norm,
            False,
            config.dropout_rates,
        )

        self.mlp2 = MlpLayer(
            self.hidden_dim,
            config.mlp2_hidden_units,
            config.batch_norm,
            False,
            config.dropout_rates,
        )

        if self.use_fs:
            self.fs_module = FeatureSelection(
                config.features,
                self.hidden_dim,
                config.embedding_dim,
                config.fs_hidden_units,
                config.fs1_context,
                config.fs2_context,
            )

        self.fusion_module = InteractionAggregation(
            config.mlp1_hidden_units[-1],
            config.mlp2_hidden_units[-1],
            output_dim=1,
            num_heads=config.num_heads,
        )
        # loss
        self.criterion = torch.nn.BCELoss()

    def add_regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "embedding" in name or "Embedding" in name:
                reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        _out = self.embedding_layer(**kwargs)

        # stack col fm
        _flat_emb = torch.stack(list(_out.values()), dim=1).flatten(start_dim=1)

        if self.use_fs:
            feat1, feat2 = self.fs_module(kwargs, _flat_emb)
        else:
            feat1, feat2 = _flat_emb, _flat_emb

        logits = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
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
