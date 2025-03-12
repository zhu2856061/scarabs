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


class CtrWithDESTINEConfig(PretrainedConfig):
    model_type = "CtrWithDESTINE"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        attention_dim=16,
        num_heads=2,
        attention_layers=2,
        dnn_hidden_units=[],
        relu_before_att=False,
        use_scale=False,
        residual_mode="each_layer",  # ['last_layer', 'each_layer', None]
        dropout_rates=0,
        batch_norm=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.attention_layers = attention_layers
        self.dnn_hidden_units = dnn_hidden_units
        self.relu_before_att = relu_before_att
        self.use_scale = use_scale
        self.residual_mode = residual_mode

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

        return torch.sum(torch.concat(_inputs, dim=-1), dim=-1, keepdim=True)


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


class DisentangledSelfAttention(torch.nn.Module):
    """Disentangle self-attention for DESTINE. The implementation is a bit different from what is
    described in the paper, but exactly follows the code from the authors:
    https://github.com/CRIPAC-DIG/DESTINE/blob/c68e182aa220b444df73286e5e928e8a072ba75e/layers/activation.py#L90
    """

    def __init__(
        self,
        embedding_dim,
        attention_dim=64,
        num_heads=1,
        dropout_rate=0.1,
        use_residual=True,
        use_scale=False,
        relu_before_att=False,
    ):
        super(DisentangledSelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.relu_before_att = relu_before_att

        self.W_q = torch.nn.Linear(embedding_dim, self.attention_dim, bias=False)
        self.W_k = torch.nn.Linear(embedding_dim, self.attention_dim, bias=False)
        self.W_v = torch.nn.Linear(embedding_dim, self.attention_dim, bias=False)
        self.W_unary = torch.nn.Linear(embedding_dim, num_heads, bias=False)

        if use_residual:
            self.W_res = torch.nn.Linear(embedding_dim, self.attention_dim)
        else:
            self.W_res = None
        self.dropout = torch.nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self.W_q.weight)
        torch.nn.init.xavier_normal_(self.W_k.weight)
        torch.nn.init.xavier_normal_(self.W_v.weight)
        torch.nn.init.xavier_normal_(self.W_unary.weight)
        if self.W_res is not None:
            torch.nn.init.xavier_normal_(self.W_res.weight)

    def forward(self, query, key, value):
        residual = query
        unary = self.W_unary(key)  # [batch, num_fields, num_heads]
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        if self.relu_before_att:
            query = query.relu()
            key = key.relu()
            value = value.relu()

        # split heads to [batch * num_heads, num_fields, head_dim]
        batch_size = query.size(0)
        query = torch.cat(query.split(split_size=self.head_dim, dim=2), dim=0)
        key = torch.cat(key.split(split_size=self.head_dim, dim=2), dim=0)
        value = torch.cat(value.split(split_size=self.head_dim, dim=2), dim=0)

        # whiten
        mu_query = query - query.mean(dim=1, keepdim=True)
        mu_key = key - key.mean(dim=1, keepdim=True)
        pair_weights = torch.bmm(mu_query, mu_key.transpose(1, 2))
        if self.use_scale:
            pair_weights /= self.head_dim**0.5
        pair_weights = F.softmax(
            pair_weights, dim=2
        )  # [num_heads * batch, num_fields, num_fields]

        unary_weights = F.softmax(unary, dim=1)
        unary_weights = unary_weights.view(batch_size * self.num_heads, -1, 1)
        unary_weights = unary_weights.transpose(
            1, 2
        )  # [num_heads * batch, 1, num_fields]

        attn_weights = pair_weights + unary_weights
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, value)
        output = torch.cat(output.split(batch_size, dim=0), dim=2)

        if self.W_res is not None:
            output += self.W_res(residual)
        return output


class CtrWithDESTINE(PreTrainedModel):
    """
    模型：DESTINE
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
     |              |             |
     LR            attns          DNN
     |_____________|______________|
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
        self.dnn_out = torch.nn.Linear(config.dnn_hidden_units[-1], 1)

        self.logitistic_layer = LogitisticLayer(config.features)

        self.self_attns = torch.nn.ModuleList(
            [
                DisentangledSelfAttention(
                    config.embedding_dim if i == 0 else config.attention_dim,
                    config.attention_dim,
                    config.num_heads,
                    config.dropout_rates,
                    config.residual_mode == "each_layer",
                    config.use_scale,
                    config.relu_before_att,
                )
                for i in range(config.attention_layers)
            ]
        )

        self.attn_out = torch.nn.Linear(self.num_fields * config.attention_dim, 1)
        if config.residual_mode == "last_layer":
            self.W_res = torch.nn.Linear(config.embedding_dim, config.attention_dim)
        else:
            self.W_res = None

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_uniform_(self.dnn_out.weight)
        torch.nn.init.zeros_(self.dnn_out.bias)

        torch.nn.init.xavier_uniform_(self.attn_out.weight)
        torch.nn.init.zeros_(self.attn_out.bias)

        if self.W_res is not None:
            torch.nn.init.xavier_uniform_(self.W_res.weight)

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

        cross_X = _out
        for self_attn in self.self_attns:
            cross_X = self_attn(cross_X, cross_X, cross_X)
        if self.W_res is not None:
            cross_X += self.W_res(_out)
        _attn_out = self.attn_out(cross_X.flatten(start_dim=1))

        _dnn_out = self.dnn_layer(_out.flatten(start_dim=1))
        _dnn_out = self.dnn_out(_dnn_out)

        _lr_out = self.logitistic_layer(**kwargs)

        logits = _attn_out + _dnn_out + _lr_out

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
