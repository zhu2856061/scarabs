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


class CtrWithFWFMConfig(PretrainedConfig):
    model_type = "CtrWithFWFM"

    def __init__(
        self,
        features=None,
        label_name="label",
        hidden_dim=8,
        linear_type="FiLV",
        regularizer=5e-06,
        **kwargs,
    ):
        """
        linear_type: `LW`, `FeLV`, or `FiLV`
        """
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.hidden_dim = hidden_dim
        self.linear_type = linear_type
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


class LW_Fifv_FelvLayer(torch.nn.Module):
    def __init__(self, features, embedding_size) -> None:
        super().__init__()

        self.features = features

        self.feature_w = torch.nn.ModuleDict()
        for name, feature in self.features.items():
            if feature["shared_embed_name"] is not None:
                continue
            self.feature_w[name] = EmbeddingUnitLayer(
                len(feature["vocab"]), embedding_size
            )

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

        return torch.concat(_inputs, dim=1)


class CtrWithFWFM(PreTrainedModel):
    """
    模型：fwfm
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
                    |
    sum(W1*H1*W2H2, W1*H1*W3H3, W1H1*W4H4, W2*H2*W3H3, W2*H2*W4H4, W3*H3*W4H4)
                    |
                   out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.linear_type = config.linear_type

        num_fields = len(config.features)
        self.interact_dim = int(num_fields * (num_fields - 1) / 2)

        self.interaction_weight_layer = torch.nn.Linear(self.interact_dim, 1)

        self.register_buffer(
            "triu_index",
            torch.triu(torch.ones(num_fields, num_fields), 1).bool(),
        )

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.hidden_dim)

        if self.linear_type == "LW":
            self.lw_fifv_felv_layer = LW_Fifv_FelvLayer(config.features, 1)

        elif self.linear_type == "FiLV":
            self.lw_fifv_felv_layer = torch.nn.Linear(
                num_fields * config.hidden_dim, 1, bias=False
            )

        elif self.linear_type == "FeLV":
            self.lw_fifv_felv_layer = LW_Fifv_FelvLayer(
                config.features, config.hidden_dim
            )

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self.interaction_weight_layer.weight)
        if self.linear_type == "FiLV":
            torch.nn.init.xavier_normal_(self.lw_fifv_felv_layer.weight)

    def add_regularization(self):
        reg_loss = 0
        for _, param in self.named_parameters():
            reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        _out_emb = self.embedding_layer(**kwargs)
        _out_emb = torch.stack(list(_out_emb.values()), dim=1)

        # stack col fwfm
        _out = torch.bmm(_out_emb, _out_emb.transpose(1, 2))

        _out = torch.masked_select(_out, self.triu_index)
        _out = _out.view(-1, self.interact_dim)
        _out = self.interaction_weight_layer(_out).sum(1)

        # linear
        if self.linear_type == "LW":
            _out_linear = self.lw_fifv_felv_layer(**kwargs)
            _out_linear = _out_linear.sum(1)
        elif self.linear_type == "FiLV":
            _out_linear = self.lw_fifv_felv_layer(_out_emb.flatten(start_dim=1))
        elif self.linear_type == "FeLV":
            _out_linear = self.lw_fifv_felv_layer(**kwargs)
            _out_linear = (_out_emb.flatten(start_dim=1) * _out_linear).sum(1)

        logits = _out + _out_linear  # bias added in self.interaction_weight_layer

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
