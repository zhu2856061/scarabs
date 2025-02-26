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


class CtrWithMaskNetConfig(PretrainedConfig):
    model_type = "CtrWithMaskNet"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        dnn_hidden_units=[64, 64, 64],
        model_mode="SerialMaskNet",
        parallel_num_blocks=1,
        parallel_block_dim=64,
        reduction_ratio=1,
        layernorm=True,
        dropout_rates=0.5,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.model_mode = model_mode
        self.parallel_num_blocks = parallel_num_blocks
        self.parallel_block_dim = parallel_block_dim
        self.reduction_ratio = reduction_ratio
        self.layernorm = layernorm
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


class MaskBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        reduction_ratio=1,
        dropout_rate=0,
        layer_norm=True,
    ):
        super(MaskBlock, self).__init__()
        self.mask_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim),
        )
        self.hidden_layer = torch.nn.Sequential()
        self.hidden_layer.add_module(
            "Linear", torch.nn.Linear(hidden_dim, output_dim, bias=False)
        )
        if layer_norm:
            self.hidden_layer.add_module("LayerNorm", torch.nn.LayerNorm(output_dim))

        self.hidden_layer.add_module("ReLU", torch.nn.ReLU())

        if dropout_rate > 0:
            self.hidden_layer.add_module("Dropout", torch.nn.Dropout(dropout_rate))

    def forward(self, V_emb, V_hidden):
        V_mask = self.mask_layer(V_emb)
        v_out = self.hidden_layer(V_mask * V_hidden)
        return v_out


class SerialMaskNet(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=[],
        reduction_ratio=1,
        dropout_rates=0,
        layer_norm=True,
    ):
        super(SerialMaskNet, self).__init__()

        self.hidden_units = [input_dim] + hidden_units
        self.mask_blocks = torch.nn.ModuleList()
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(
                MaskBlock(
                    input_dim,
                    self.hidden_units[idx],
                    self.hidden_units[idx + 1],
                    reduction_ratio,
                    dropout_rates,
                    layer_norm,
                )
            )

    def forward(self, V_emb, V_hidden):
        v_out = V_hidden
        for idx in range(len(self.hidden_units) - 1):
            v_out = self.mask_blocks[idx](V_emb, v_out)
        return v_out


class ParallelMaskNet(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        num_blocks=1,
        block_dim=64,
        hidden_units=[64, 64],
        reduction_ratio=1,
        dropout_rates=0,
        layer_norm=True,
    ):
        super(ParallelMaskNet, self).__init__()
        self.num_blocks = num_blocks
        self.mask_blocks = torch.nn.ModuleList(
            [
                MaskBlock(
                    input_dim,
                    input_dim,
                    block_dim,
                    reduction_ratio,
                    dropout_rates,
                    layer_norm,
                )
                for _ in range(num_blocks)
            ]
        )

        self.dnn = MlpLayer(
            input_dim=block_dim * num_blocks,
            hidden_units=hidden_units,
            batch_norm=False,
            dropout_rates=dropout_rates,
        )

    def forward(self, V_emb, V_hidden):
        block_out = []
        for i in range(self.num_blocks):
            block_out.append(self.mask_blocks[i](V_emb, V_hidden))
        concat_out = torch.cat(block_out, dim=-1)
        v_out = self.dnn(concat_out)
        return v_out


class CtrWithMaskNet(PreTrainedModel):
    """
    模型：SerialMask
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
                    |
                MaskBlock
                    |
                MaskBlock
                    |
                MaskBlock
                    |
                   out

    结构：ParallelMask
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
        |           |             |
    MaskBlock   MaskBlock ... MaskBlock
                    |
                   DNN
                    |
                   out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.num_fields = len(config.features)
        self.model_mode = config.model_mode
        self.layernorm = config.layernorm
        self.hidden_dim = self.num_fields * config.embedding_dim
        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)

        if self.model_mode == "SerialMaskNet":
            self.mask_net = SerialMaskNet(
                input_dim=self.num_fields * config.embedding_dim,
                hidden_units=config.dnn_hidden_units,
                reduction_ratio=config.reduction_ratio,
                dropout_rates=config.dropout_rates,
                layer_norm=config.layernorm,
            )

        elif self.model_mode == "ParallelMaskNet":
            self.mask_net = ParallelMaskNet(
                input_dim=self.num_fields * config.embedding_dim,
                num_blocks=config.parallel_num_blocks,
                block_dim=config.parallel_block_dim,
                hidden_units=config.dnn_hidden_units,
                reduction_ratio=config.reduction_ratio,
                dropout_rates=config.dropout_rates,
                layer_norm=config.layernorm,
            )

        if self.layernorm:
            self.emb_norm = torch.nn.ModuleList(
                torch.nn.LayerNorm(config.embedding_dim) for _ in range(self.num_fields)
            )
        else:
            self.emb_norm = None

        self.out = torch.nn.Linear(config.dnn_hidden_units[-1], 1)

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

        if self.emb_norm:
            _out_list = _out.chunk(self.num_fields, dim=1)
            V_hidden = torch.cat(
                [self.emb_norm[i](feat) for i, feat in enumerate(_out_list)], dim=1
            )
        else:
            V_hidden = _out

        logits = self.mask_net(
            V_hidden.flatten(start_dim=1), V_hidden.flatten(start_dim=1)
        )

        logits = self.out(logits)
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
