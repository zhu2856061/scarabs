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
class RecallModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class RecallWithDCLConfig(PretrainedConfig):
    model_type = "CtrWithDCL"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_size=8,
        user_features=[],
        user_tower_units=[64, 64, 64],
        item_tower_units=[64, 64, 64],
        all_bt_coeff=1,
        poly_coeff=0.2,
        mom_coeff=5,
        bt_coeff=0.01,
        a=1,
        polyc=1e-7,
        degree=4,
        momentum=0.1,
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

        self.all_bt_coeff = all_bt_coeff
        self.poly_coeff = poly_coeff
        self.mom_coeff = mom_coeff
        self.bt_coeff = bt_coeff
        self.a = a
        self.polyc = polyc
        self.degree = degree
        self.momentum = momentum

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


class RecallWithDCL(PreTrainedModel):
    """
    模型：Dual Contrastive Learning for Recommendation
    由于模型的结构可以采用任何其他模型，而对比学习的本质是loss层级的修改
    故，这里的核心在于 FCL BCL 的组合
    结构：
    u_fea1     u_fea2      i_fea3     i_fea4
     |         |              |        |
     H         H              H        H
     |_________|              |________|
          |                        |
         DNN                      DNN
          |                        |
    user_embedding            item_embedding


    FCL: Luibt = 1/F * sum((1 - Cmm)^2) + bt_coeff * 1/F * sum(Cmn^2)
         UUII = ( a * (u_emb ^T * u_emb) + polyc )^degree + (a * (i_emb ^T * i_emb) + polyc )^degree

    BCL: Luibt = 1/F * sim(u_emb, i_his_emb) - 1/F * sim(i_emb, u_his_emb)

    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.num_fields = len(config.features)
        self.user_features = config.user_features

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_size)

        # user tower
        self.user_tower_layer = MlpLayer(
            len(self.user_features) * config.embedding_size,
            config.user_tower_units[:-1],
            config.batch_norm,
            config.dropout_rates,
        )
        self.out_user = torch.nn.Linear(
            config.user_tower_units[-2], config.user_tower_units[-1]
        )

        # item tower
        self.item_tower_layer = MlpLayer(
            (self.num_fields - len(self.user_features)) * config.embedding_size,
            config.item_tower_units[:-1],
            config.batch_norm,
            config.dropout_rates,
        )
        self.out_item = torch.nn.Linear(
            config.item_tower_units[-2], config.item_tower_units[-1]
        )

        # loss parameters
        self.all_bt_coeff = config.all_bt_coeff
        self.poly_coeff = config.poly_coeff
        self.mom_coeff = config.mom_coeff

        self.bt_coeff = config.bt_coeff
        self.a = config.a
        self.polyc = config.polyc
        self.degree = config.degree
        self.momentum = config.momentum

        self.uid = "uid"
        self.iid = "iid"
        self.n_users = len(config.features["uid"]["vocab"])
        self.n_items = len(config.features["iid"]["vocab"])

        self.register_buffer(
            "u_target_his",
            torch.randn(
                (self.n_users, config.item_tower_units[-1]), requires_grad=False
            ),
        )

        self.register_buffer(
            "i_target_his",
            torch.randn(
                (self.n_items, config.item_tower_units[-1]), requires_grad=False
            ),
        )

        # loss
        self.bn = torch.nn.BatchNorm1d(config.item_tower_units[-2], affine=False)
        self.criterion = torch.nn.BCELoss()

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
        user = kwargs[self.uid]
        item = kwargs[self.iid]
        _embedding_layer = self.embedding_layer(**kwargs)

        _user = []
        _item = []
        for k, v in _embedding_layer.items():
            if k in self.user_features:
                _user.append(v)
            else:
                _item.append(v)

        # stack col fm
        _user = torch.concat(_user, dim=-1)
        _item = torch.concat(_item, dim=-1)

        _user_encoder = self.out_user(self.user_tower_layer(_user))
        _item_encoder = self.out_item(self.item_tower_layer(_item))

        with torch.no_grad():
            # 扰动因子，his embedding
            user_res = user.squeeze(1)
            item_res = item.squeeze(1)
            u_target = self.u_target_his[user_res, :]
            i_target = self.i_target_his[item_res, :]

            u_target = u_target * self.momentum + _user_encoder * (1.0 - self.momentum)
            i_target = i_target * self.momentum + _item_encoder * (1.0 - self.momentum)

            self.u_target_his[user_res, :] = u_target
            self.i_target_his[item_res, :] = i_target

        _user_encoder_n = F.normalize(_user_encoder, dim=-1)
        _item_encoder_n = F.normalize(_item_encoder, dim=-1)

        loss_uibt = self.LUIBT(_user_encoder_n, _item_encoder_n)
        loss_uuii = self.UUII(_user_encoder_n, _item_encoder_n)
        loss_bcl = self.LBCL(_user_encoder, _item_encoder, u_target, i_target)

        loss = (
            self.all_bt_coeff * loss_uibt
            + self.poly_coeff * loss_uuii
            + self.mom_coeff * loss_bcl
        )

        logits = torch.sigmoid(torch.sum(_user_encoder_n * _item_encoder_n, dim=-1))
        return RecallModelOutput(loss=loss, logits=logits)

    def LUIBT(self, user_embedding, item_embedding):
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        b, e = user_embedding.size()
        c = self.bn(user_embedding).T @ self.bn(item_embedding)
        c.div_(b)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(e)
        off_diag = off_diagonal(c).pow_(2).sum().div(e)
        bt_loss = on_diag + self.bt_coeff * off_diag
        return bt_loss

    def UUII(self, user_embedding, item_embedding):
        user_e = self.bn(user_embedding).T @ self.bn(user_embedding)
        item_e = self.bn(item_embedding).T @ self.bn(item_embedding)

        poly_user_e = (self.a * user_e + self.polyc) ** self.degree
        poly_user_e = poly_user_e.mean().log()

        poly_item_e = (self.a * item_e + self.polyc) ** self.degree
        poly_item_e = poly_item_e.mean().log()

        poly_loss = poly_user_e / 2 + poly_item_e / 2
        return poly_loss

    def LBCL(self, user_embedding, item_embedding, u_target, i_target):
        l1 = -F.cosine_similarity(user_embedding, i_target, dim=-1).mean()
        l2 = -F.cosine_similarity(item_embedding, u_target, dim=-1).mean()

        return l1 / 2 + l2 / 2

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
