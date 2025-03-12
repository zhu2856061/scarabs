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

from scarabs.mora.utils.loss import BPRLoss


@dataclass
class RecallModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class RecallWithDSSMConfig(PretrainedConfig):
    model_type = "RecallWithDSSM"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_size=8,
        user_features=[],
        user_tower_units=[64, 64, 64],
        item_tower_units=[64, 64, 64],
        dropout_rates=0,
        batch_norm=False,
        regularizer=5e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_size = embedding_size
        self.user_features = user_features
        self.user_tower_units = user_tower_units
        self.item_tower_units = item_tower_units

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
            if name not in kwargs:
                continue
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


class RecallWithDSSM(PreTrainedModel):
    """
    模型：DSSM

    结构：
    u_fea1     u_fea2      i_fea3     i_fea4
     |         |              |        |
     H         H              H        H
     |_________|              |________|
          |                        |
         DNN                      DNN
          |                        |
    user_embedding     *      item_embedding

    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.num_fields = len(config.features)
        self.user_features = config.user_features
        self.n_items = len(config.features["iid"]["vocab"])

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_size)

        # user tower
        self.user_tower_layer = MlpLayer(
            len(self.user_features) * config.embedding_size,
            config.user_tower_units,
            config.batch_norm,
            config.dropout_rates,
        )
        self.out_user = torch.nn.Linear(
            config.user_tower_units[-2], config.user_tower_units[-1]
        )

        # item tower
        self.item_tower_layer = MlpLayer(
            (self.num_fields - len(self.user_features)) * config.embedding_size,
            config.item_tower_units,
            config.batch_norm,
            config.dropout_rates,
        )
        self.out_item = torch.nn.Linear(
            config.item_tower_units[-2], config.item_tower_units[-1]
        )

        # define item embs matrix
        self.register_buffer(
            "item_tower_embs",
            torch.zeros(
                (self.n_items, config.item_tower_units[-1]), requires_grad=False
            ),
        )
        # loss
        self.criterion = BPRLoss()

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self.out_user.weight)
        torch.nn.init.zeros_(self.out_user.bias)

        torch.nn.init.xavier_normal_(self.out_item.weight)
        torch.nn.init.zeros_(self.out_item.bias)

    def add_regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "embedding" in name or "Embedding" in name:
                reg_loss += (self.regularizer / 2) * torch.norm(param, 2) ** 2

        return reg_loss

    def forward(self, **kwargs):
        """
        确保输入的格式如下
        {
         'uids': {'uid': tensor([[11781],[ 1488]])},
         'pos_iids': {'iid': tensor([[14500],[  567]])},
         'neg_iids': {'iid': tensor([[11789],[30130]])}
        }

        uids 为用户侧特征
        pos_iids 为item正样本侧特征
        neg_iids 为item负样本侧特征
        """
        _uids = self.embedding_layer(**kwargs["uids"])
        _pos_iids = self.embedding_layer(**kwargs["pos_iids"])

        _pos_item_index = kwargs["pos_iids"]["iid"].squeeze(1)
        _neg_iids = self.embedding_layer(**kwargs["neg_iids"])
        _neg_item_index = kwargs["neg_iids"]["iid"].squeeze(1)

        # stack col
        _uids = torch.concat(list(_uids.values()), dim=-1)
        _pos_iids = torch.concat(list(_pos_iids.values()), dim=-1)
        _neg_iids = torch.concat(list(_neg_iids.values()), dim=-1)

        # in tower
        _uids = self.out_user(self.user_tower_layer(_uids))
        _pos_iids = self.out_item(self.item_tower_layer(_pos_iids))
        _neg_iids = self.out_item(self.item_tower_layer(_neg_iids))

        self.item_tower_embs[_pos_item_index, :] = _pos_iids
        self.item_tower_embs[_neg_item_index, :] = _neg_iids

        pos_logits = (_uids * _pos_iids).sum(dim=-1)
        neg_logits = (_uids * _neg_iids).sum(dim=-1)

        pos_logits = pos_logits.contiguous().view(-1)
        neg_logits = neg_logits.contiguous().view(-1)
        loss = self.criterion(pos_logits, neg_logits) + self.add_regularization()

        return RecallModelOutput(loss=loss)

    def get_user_embedding(self, **user_kwargs):
        _embedding_layer = self.embedding_layer(**user_kwargs)
        _user = []
        for k, v in _embedding_layer.items():
            if k in self.user_features:
                _user.append(v)

        _user = torch.concat(_user, dim=-1)
        _user_encoder = self.out_user(self.user_tower_layer(_user))
        return _user_encoder

    def get_item_embedding(self, **item_kwargs):
        _embedding_layer = self.embedding_layer(**item_kwargs)
        _item = []
        for k, v in _embedding_layer.items():
            if k not in self.user_features:
                _item.append(v)

        _item = torch.concat(_item, dim=-1)
        _item_encoder = self.out_item(self.item_tower_layer(_item))
        return _item_encoder

    def get_item_matrix(self):
        return self.item_tower_embs
