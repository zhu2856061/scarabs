# -*- coding: utf-8 -*-
# @Time   : 2024/08/26 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import json
import os
from collections import OrderedDict
from typing import Optional

import numpy as np
from loguru import logger


# define the structure of Features
class Feature:
    def __init__(self, **kwargs):
        self.target = kwargs.get("target")

        self.name = kwargs.get("name")
        self.length = kwargs.get("length")
        self.default = kwargs.get("default")
        self.embed_size = kwargs.get("embed_size")
        self.shared_embed_name = kwargs.get("share_embed_name")
        self.vocab = kwargs.get("vocab")

        self.minv = kwargs.get("minv")
        self.maxv = kwargs.get("maxv")
        self.sep = kwargs.get("sep")

        # handle
        if self.target is None:
            raise ValueError(">>>>> feature target is None <<<<<")
        if self.name is None:
            raise ValueError(">>>>> feature name is None <<<<<")
        if self.length is None:
            raise ValueError(">>>>> feature length is None <<<<<")
        if self.default is None:
            raise ValueError(">>>>> feature default is None <<<<<")

        if self.embed_size is None:
            logger.warning(
                f"{self.name} embedding_size is None, will be set from model"
            )

        if self.minv is not None and self.maxv is not None and self.sep is not None:
            self.step = (self.maxv - self.minv) / self.sep

        # give default value to initial input into vacab
        if self.vocab is None:
            self.vocab = {self._tranfrom_value(self.default): 0}

    # init meta
    def init_vocab_meta(self, value):
        if value is not None and self.vocab is not None:
            if not isinstance(value, list):
                value = [value]
            for v in value:
                v = self._tranfrom_value(v)
                try:
                    _ = self.vocab[v]
                except KeyError:
                    self.vocab[v] = len(self.vocab)

    # batch init meta
    def init_vocab_meta_batch(self, values):
        for value in values:
            self.init_vocab_meta(value)

    # unified conversion Tool
    def _tranfrom_value(self, value: Optional[str | float | int]):
        raise NotImplementedError

    # convert the input value
    def handle(self, value):
        if self.vocab is None:
            raise ValueError(">>>>> feature vocab is None <<<<<")
        if not isinstance(self.length, int):
            raise ValueError(">>>>> feature length is error <<<<<")

        if not isinstance(value, list):
            value = [value]
        new_value = []
        for v in value:
            v = self._tranfrom_value(v)
            try:
                v = self.vocab[v]
            except KeyError:
                v = self._tranfrom_value(self.default)
                v = self.vocab[v]

            new_value.append(v)

        if len(new_value) > self.length:
            new_value = new_value[: self.length]
        else:
            new_value = new_value + [0] * (self.length - len(new_value))
        return new_value

    # batch convert the input value
    def batch_handle(self, values):
        new_value = [self.handle(value) for value in values]
        return new_value


# convert character based data or arrays into Features
class Hash2Feature(Feature):
    def _tranfrom_value(self, value):
        if self.name is None:
            raise ValueError(">>>>> feature name is None <<<<<")
        if self.default is None:
            raise ValueError(">>>>> feature default is None <<<<<")

        if not isinstance(self.default, str):
            raise ValueError(">>>>> feature default is error <<<<<")

        if value is None:
            value = self.default

        return self.name + "_" + str(value)


# bucket numerical data and convert it into Features
class MinMaxBucket2Feature(Feature):
    def _tranfrom_value(self, value):
        if self.name is None:
            raise ValueError(">>>>> feature name is None <<<<<")
        if self.minv is None:
            raise ValueError(">>>>> feature minv is None <<<<<")
        if self.maxv is None:
            raise ValueError(">>>>> feature maxv is None <<<<<")
        if self.sep is None:
            raise ValueError(">>>>> feature sep is None <<<<<")

        if value is None:
            value = self.default

        if isinstance(self.default, str):
            raise ValueError(">>>>> feature default is error <<<<<")

        if value <= self.minv:
            value = 0
        elif value >= self.maxv:
            value = self.sep - 1
        else:
            value = (value - self.minv) // self.sep

        return self.name + "_" + str(int(value))


# convert numerical data into an index by performing floor-log bucket partitioning on it
class FloorLogBucket2Feature(Feature):
    def _tranfrom_value(self, value):
        # Set 0 for values less than or equal to 1
        # Log and lower bound for values greater than 1
        if self.name is None:
            raise ValueError(">>>>> feature name is None <<<<<")

        if value is None:
            value = self.default

        if isinstance(self.default, str):
            raise ValueError(">>>>> feature default is error <<<<<")

        if value <= 1:  # type: ignore
            value = 0
        if value > 1:  # type: ignore
            value = int(np.log(value))  # type: ignore

        return self.name + "_" + str(value)


# For table data, convert it into the input format required by the model
class Feature2Transformer:
    def __init__(self, meta_dir: str = "./meta"):
        self.meta_dir = meta_dir
        self.feature2meta = OrderedDict()

    def create_meta(self, params):
        for item in params:
            name = item["name"]
            if item["target"] == "Hash2Feature":
                obj = Hash2Feature(**item)
                self.feature2meta[name] = obj
            if item["target"] == "MinMaxBucket2Feature":
                obj = MinMaxBucket2Feature(**item)
                self.feature2meta[name] = obj
            if item["target"] == "FloorLogBucket2Feature":
                obj = FloorLogBucket2Feature(**item)
                self.feature2meta[name] = obj

    def build_meta(self, example):
        for name, fea in self.feature2meta.items():
            if fea.shared_embed_name is None:
                if name in example:
                    fea.init_vocab_meta(example[name])
            else:
                if name in example:
                    self.feature2meta[fea.shared_embed_name].init_vocab_meta(
                        example[name]
                    )

    def build_meta_batch(self, examples):
        for name, fea in self.feature2meta.items():
            if fea.shared_embed_name is None:
                if name in examples:
                    fea.init_vocab_meta_batch(examples[name])
            else:
                if name in examples:
                    self.feature2meta[fea.shared_embed_name].init_vocab_meta_batch(
                        examples[name]
                    )

    def handle(self, example, selected_columns=None):
        new_example = {}
        for name, fea in self.feature2meta.items():
            if selected_columns is not None and name not in selected_columns:
                continue
            tmp = example.get(name, None)
            if fea.shared_embed_name is None:
                new_example[name] = fea.handle(tmp)
            else:
                new_example[name] = self.feature2meta[fea.shared_embed_name].handle(tmp)
        return new_example

    def save_meta(self):
        # 转json
        obj_dict = {}
        for name, fea in self.feature2meta.items():
            obj_dict[name] = fea.__dict__

        os.makedirs(self.meta_dir, exist_ok=True)
        with open(os.path.join(self.meta_dir, "feature2meta.json"), "w") as f:
            json.dump(obj_dict, f)

    def load_meta(self, config=None):
        if config is None:
            if not os.path.exists(os.path.join(self.meta_dir, "feature2meta.json")):
                raise ValueError(f">>>>> file: {self.meta_dir} not exists <<<<<")

            with open(os.path.join(self.meta_dir, "feature2meta.json"), "r") as f:
                feature2meta_json = json.load(f)
                for item, value in feature2meta_json.items():
                    if value["target"] == "Hash2Feature":
                        self.feature2meta[item] = Hash2Feature(**value)
                    if value["target"] == "MinMaxBucket2Feature":
                        self.feature2meta[item] = MinMaxBucket2Feature(**value)
                    if value["target"] == "FloorLogBucket2Feature":
                        self.feature2meta[item] = FloorLogBucket2Feature(**value)

                assert len(self.feature2meta) == len(
                    feature2meta_json
                ), ">>>>> feature2meta unmatched <<<<<"

        else:
            if config.features is None:
                raise ValueError(">>>>> config.features is None <<<<<")
            for item, value in config.features.items():
                if value["target"] == "Hash2Feature":
                    self.feature2meta[item] = Hash2Feature(**value)
                if value["target"] == "MinMaxBucket2Feature":
                    self.feature2meta[item] = MinMaxBucket2Feature(**value)
                if value["target"] == "FloorLogBucket2Feature":
                    self.feature2meta[item] = FloorLogBucket2Feature(**value)

            assert len(self.feature2meta) == len(
                config.features
            ), ">>>>> feature2meta unmatched <<<<<"
