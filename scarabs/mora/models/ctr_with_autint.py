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


class CtrWithAutoIntConfig(PretrainedConfig):
    model_type = "CtrWithAutoInt"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        dnn_hidden_units=[64, 64, 64],
        attention_layers=2,
        num_heads=1,
        attention_dim=8,
        layer_norm=False,
        batch_norm=True,
        use_scale=False,
        use_residual=True,
        dropout_rates=0.1,
        regularizer=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.attention_layers = attention_layers
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.use_scale = use_scale
        self.use_residual = use_residual

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


class ScaledDotProductAttention(torch.nn.Module):
    """Scaled Dot-Product Attention
    Ref: https://zhuanlan.zhihu.com/p/47812375
    """

    def __init__(self, dropout_rate=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, scale=None, mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask:
            scores = scores.masked_fill_(mask, -1e-10)
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention


class MultiHeadSelfAttention(torch.nn.Module):
    """Multi-head attention module"""

    def __init__(
        self,
        input_dim,
        attention_dim=None,
        num_heads=1,
        dropout_rate=0.0,
        use_residual=True,
        use_scale=False,
        layer_norm=False,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert (
            attention_dim % num_heads == 0
        ), "attention_dim={} is not divisible by num_heads={}".format(
            attention_dim, num_heads
        )
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim**0.5 if use_scale else None
        self.W_q = torch.nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = torch.nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = torch.nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = torch.nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(attention_dim) if layer_norm else None

        self._init_weight_()

    def _init_weight_(self):
        """Fine tune details: init weight very important !!!"""
        torch.nn.init.xavier_normal_(self.W_q.weight)
        torch.nn.init.xavier_normal_(self.W_k.weight)
        torch.nn.init.xavier_normal_(self.W_v.weight)
        if self.W_res is not None:
            torch.nn.init.xavier_normal_(self.W_res.weight)

    def forward(self, X):
        residual = X

        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        # concat heads
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )

        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output


class CtrWithAutoInt(PreTrainedModel):
    """
    模型：afn
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
            |               |     |
    multi-head-attention   dnn    lr
            |_______________|_____|
                    |
                   out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        if config.features is None:
            raise ValueError("config features must be provided")
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.num_fields = len(config.features)

        # define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)
        self.lr_layer = LogitisticLayer(config.features)
        self.mlp_layer = MlpLayer(
            config.embedding_dim * self.num_fields,
            config.dnn_hidden_units,
            config.batch_norm,
            config.dropout_rates,
        )
        self.mlp_layer_out = torch.nn.Linear(config.dnn_hidden_units[-1], 1)

        self.self_attention = torch.nn.Sequential(
            *[
                MultiHeadSelfAttention(
                    config.embedding_dim if i == 0 else config.attention_dim,
                    attention_dim=config.attention_dim,
                    num_heads=config.num_heads,
                    dropout_rate=config.dropout_rates,
                    use_residual=config.use_residual,
                    use_scale=config.use_scale,
                    layer_norm=config.layer_norm,
                )
                for i in range(config.attention_layers)
            ]
        )
        # out
        self.out = torch.nn.Linear(self.num_fields * config.attention_dim, 1)

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
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

        _attention_out = self.self_attention(_inputs)
        _attention_out = torch.flatten(_attention_out, start_dim=1)

        _out = self.out(_attention_out)

        _lr_out = self.lr_layer(**kwargs)

        _dnn_out = self.mlp_layer(_inputs.flatten(start_dim=1))
        _dnn_out = self.mlp_layer_out(_dnn_out)

        logits = torch.sigmoid(_out.view(-1) + _lr_out.view(-1) + _dnn_out.view(-1))

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
