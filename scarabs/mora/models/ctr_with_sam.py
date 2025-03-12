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


class CtrWithSAMConfig(PretrainedConfig):
    model_type = "CtrWithSAM"

    def __init__(
        self,
        features=None,
        label_name="label",
        embedding_dim=8,
        interaction_type="SAM2E",  # option in ["SAM2A", "SAM2E", "SAM3A", "SAM3E"]
        aggregation="concat",
        num_interaction_layers=3,
        use_residual=False,
        dropout_rates=0.1,
        regularizer=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.embedding_dim = embedding_dim
        self.interaction_type = interaction_type
        self.aggregation = aggregation
        self.num_interaction_layers = num_interaction_layers
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


class SAMBlock(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        num_fields,
        embedding_dim,
        use_residual=False,
        interaction_type="SAM2E",
        aggregation="concat",
        dropout=0,
    ):
        super(SAMBlock, self).__init__()
        assert aggregation in [
            "concat",
            "weighted_pooling",
            "mean_pooling",
            "sum_pooling",
        ]
        self.aggregation = aggregation
        if self.aggregation == "weighted_pooling":
            self.weight = torch.nn.Parameter(torch.ones(num_fields, 1))
        if interaction_type == "SAM2A":
            assert (
                aggregation == "concat"
            ), "Only aggregation=concat is supported for SAM2A."
            self.layers = torch.nn.ModuleList(
                [SAM2A(num_fields, embedding_dim, dropout)]
            )
        elif interaction_type == "SAM2E":
            assert (
                aggregation == "concat"
            ), "Only aggregation=concat is supported for SAM2E."
            self.layers = torch.nn.ModuleList([SAM2E(embedding_dim, dropout)])
        elif interaction_type == "SAM3A":
            self.layers = torch.nn.ModuleList(
                [
                    SAM3A(num_fields, embedding_dim, use_residual, dropout)
                    for _ in range(num_layers)
                ]
            )
        elif interaction_type == "SAM3E":
            self.layers = torch.nn.ModuleList(
                [SAM3E(embedding_dim, use_residual, dropout) for _ in range(num_layers)]
            )
        else:
            raise ValueError(
                "interaction_type={} not supported.".format(interaction_type)
            )

    def forward(self, F):
        for layer in self.layers:
            F = layer(F)
        if self.aggregation == "concat":
            out = F.flatten(start_dim=1)
        elif self.aggregation == "weighted_pooling":
            out = (F * self.weight).sum(dim=1)
        elif self.aggregation == "mean_pooling":
            out = F.mean(dim=1)
        elif self.aggregation == "sum_pooling":
            out = F.sum(dim=1)
        return out


class SAM2A(torch.nn.Module):
    def __init__(self, num_fields, embedding_dim, dropout=0):
        super(SAM2A, self).__init__()
        self.W = torch.nn.Parameter(
            torch.ones(num_fields, num_fields, embedding_dim)
        )  # f x f x d
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2))  # b x f x f
        out = S.unsqueeze(-1) * self.W  # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM2E(torch.nn.Module):
    def __init__(self, embedding_dim, dropout=0):
        super(SAM2E, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2))  # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F)  # b x f x f x d
        out = S.unsqueeze(-1) * U  # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM3A(torch.nn.Module):
    def __init__(self, num_fields, embedding_dim, use_residual=True, dropout=0):
        super(SAM3A, self).__init__()
        self.W = torch.nn.Parameter(
            torch.ones(num_fields, num_fields, embedding_dim)
        )  # f x f x d
        self.K = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else None

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self.K.weight)
        if self.use_residual:
            torch.nn.init.xavier_normal_(self.Q.weight)

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2))  # b x f x f
        out = (S.unsqueeze(-1) * self.W).sum(dim=2)  # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM3E(torch.nn.Module):
    def __init__(self, embedding_dim, use_residual=True, dropout=0):
        super(SAM3E, self).__init__()
        self.K = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else None

        self._init_weight_()

    def _init_weight_(self):
        torch.nn.init.xavier_normal_(self.K.weight)
        if self.use_residual:
            torch.nn.init.xavier_normal_(self.Q.weight)

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2))  # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F)  # b x f x f x d
        out = (S.unsqueeze(-1) * U).sum(dim=2)  # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out


class CtrWithSAM(PreTrainedModel):
    """
    模型：afn
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
                   |
                SAMBlock-> [SAM2A | SAM2E | SAM3A | SAM3E]
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

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)

        self.block = SAMBlock(
            config.num_interaction_layers,
            self.num_fields,
            config.embedding_dim,
            config.use_residual,
            config.interaction_type,
            config.aggregation,
            config.dropout_rates,
        )

        if config.aggregation == "concat":
            if config.interaction_type in ["SAM2A", "SAM2E"]:
                self.out = torch.nn.Linear(
                    config.embedding_dim * (self.num_fields**2), 1
                )
            else:
                self.out = torch.nn.Linear(self.num_fields * config.embedding_dim, 1)
        else:
            self.out = torch.nn.Linear(config.embedding_dim, 1)

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

        _final_out = self.block(_inputs)

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
