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


class CtrWithDCNV2Config(PretrainedConfig):
    model_type = "CtrWithDCNV2"

    def __init__(
        self,
        features=None,
        label_name="label",
        model_structure="parallel",  # crossnet_only, stacked, parallel, stacked_parallel
        use_low_rank_mixture=False,
        low_rank=32,
        num_experts=4,
        embedding_dim=8,
        batch_norm=False,
        dropout_rates=0.1,
        stacked_dnn_hidden_units=[],
        parallel_dnn_hidden_units=[],
        num_cross_layers=3,
        regularizer=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.label_name = label_name
        self.model_structure = model_structure
        self.use_low_rank_mixture = use_low_rank_mixture
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        self.batch_norm = batch_norm
        self.dropout_rates = dropout_rates
        self.stacked_dnn_hidden_units = stacked_dnn_hidden_units
        self.parallel_dnn_hidden_units = parallel_dnn_hidden_units
        self.num_cross_layers = num_cross_layers
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


class CrossNetV2(torch.nn.Module):
    """
    模型：cross net v2
    结构：
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
          |                 |
      cross net v2         mlp
          |_________________|
                    |
                   out
    """

    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = torch.nn.ModuleList(
            torch.nn.Linear(input_dim, input_dim) for _ in range(self.num_layers)
        )

        self._init_weight_()

    def _init_weight_(self):
        for _ in self.cross_layers:
            torch.nn.init.xavier_normal_(_.weight)
            torch.nn.init.zeros_(_.bias)

    def forward(self, X_0):
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i


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


class CrossNetMix(torch.nn.Module):
    """CrossNetMix improves CrossNet by:
    1. add MOE to learn feature interactions in different subspaces
    2. add nonlinear transformations in low-dimensional space
    """

    def __init__(self, in_features, layer_num=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(num_experts, in_features, low_rank)
                    )
                )
                for i in range(self.layer_num)
            ]
        )
        # V: (in_features, low_rank)
        self.V_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(num_experts, in_features, low_rank)
                    )
                )
                for i in range(self.layer_num)
            ]
        )
        # C: (low_rank, low_rank)
        self.C_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(num_experts, low_rank, low_rank)
                    )
                )
                for i in range(self.layer_num)
            ]
        )
        self.gating = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features, 1, bias=False)
                for i in range(self.num_experts)
            ]
        )

        self.bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self.layer_num)
            ]
        )

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(
                    self.V_list[i][expert_id].t(), x_l
                )  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(
                    self.U_list[i][expert_id], v_x
                )  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(
                output_of_experts, 2
            )  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(
                gating_score_of_experts, 1
            )  # (bs, num_experts, 1)
            moe_out = torch.matmul(
                output_of_experts, gating_score_of_experts.softmax(1)
            )
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


class CtrWithDCNV2(PreTrainedModel):
    """
    模型：dcnv2
    结构：Parallel
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
          |                 |
      cross net v2         mlp
          |_________________|
                    |
                   out

    结构：Stack
    fea1     fea2      fea3     fea4
     |         |         |        |
     H         H         H        H
     |_________|_________|________|
                   |
             cross net v2
                   |
                  mlp
                   |
                  out
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.label_name = config.label_name
        self.regularizer = config.regularizer
        self.model_structure = config.model_structure
        self.use_low_rank_mixture = config.use_low_rank_mixture

        assert self.model_structure in [
            "crossnet_only",
            "stacked",
            "parallel",
            "stacked_parallel",
        ], "model_structure={} not supported!".format(self.model_structure)

        #  define embedding layer
        self.embedding_layer = EmbeddingLayer(config.features, config.embedding_dim)

        input_dim = len(config.features) * config.embedding_dim

        if config.use_low_rank_mixture:
            self.crossnet = CrossNetMix(
                input_dim,
                config.num_cross_layers,
                low_rank=config.low_rank,
                num_experts=config.num_experts,
            )
        else:
            self.crossnet = CrossNetV2(input_dim, config.num_cross_layers)

        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MlpLayer(
                input_dim,
                config.stacked_dnn_hidden_units,
                config.batch_norm,
                config.dropout_rates,
            )
            final_dim = config.stacked_dnn_hidden_units[-1]

        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MlpLayer(
                input_dim,
                config.parallel_dnn_hidden_units,
                config.batch_norm,
                config.dropout_rates,
            )
            final_dim = input_dim + config.parallel_dnn_hidden_units[-1]

        if self.model_structure == "stacked_parallel":
            final_dim = (
                config.stacked_dnn_hidden_units[-1]
                + config.parallel_dnn_hidden_units[-1]
            )

        if self.model_structure == "crossnet_only":  # only CrossNet
            final_dim = input_dim

        # out
        self.out = torch.nn.Linear(final_dim, 1)

        # loss
        self.criterion = torch.nn.BCELoss()

        self._init_weight_()

    def _init_weight_(self):
        # torch.nn.init.kaiming_uniform_(self.out.weight, a=1, nonlinearity="sigmoid")
        # torch.nn.init.zeros_(self.out.bias)

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

        _inputs = torch.concat(list(_inputs.values()), dim=1)
        _inputs = _inputs.flatten(start_dim=1)

        _cross_out = self.crossnet(_inputs)

        if self.model_structure == "crossnet_only":
            _final_out = _cross_out
        elif self.model_structure == "stacked":
            _final_out = self.stacked_dnn(_cross_out)
        elif self.model_structure == "parallel":
            _dnn_out = self.parallel_dnn(_inputs)
            _final_out = torch.cat([_cross_out, _dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            _final_out = torch.cat(
                [self.stacked_dnn(_cross_out), self.parallel_dnn(_inputs)],
                dim=-1,
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
