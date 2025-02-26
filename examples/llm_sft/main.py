# -*- coding: utf-8 -*-
# @Time   : 2024/09/28 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.append("../..")

from typing import Optional, Tuple, Type, Union

import datasets
import evaluate
import torch
from loguru import logger
from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)
from torchinfo import summary
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.trainer_utils import get_last_checkpoint

from scarabs.args_factory import DataArguments, ModelArguments, TrainArguments
from scarabs.data_factory import (
    DataFactoryWithSFT,
)
from scarabs.task_factory import TaskFactory, TaskFactoryWithSFTLoratrain

task = TaskFactoryWithSFTLoratrain()
# Params
parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore

model_args, data_args, training_args = parser.parse_json_file("arguments.json")

# define model
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "token": model_args.token,
    "trust_remote_code": model_args.trust_remote_code,
}
config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
model = Qwen2ForCausalLM(config)

# check model
input_ids = torch.randint(0, 10, (2, 512))
attention_mask = torch.ones_like(input_ids)
summary(
    model,
    input_data={"input_ids": input_ids, "attention_mask": attention_mask},
    depth=5,
)

# train
task.train(model_args, data_args, training_args, model=model, config=config)

# # predict
# task = TaskFactoryWithPreTrain()
# task.inference_with_load_model(
#     "../data/llama-160m",
#     "../data/llama-160m",
#     Qwen2ForCausalLM,
# )
# res = task.inference("布偶猫", max_tokens=2000)
# print(res)
# accelerate launch --num_processes=2 main.py
# torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py arguments.json
# accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} main.py --all_arguments_of_the_script
