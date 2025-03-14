# -*- coding: utf-8 -*-
# @Time   : 2024/07/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import dataclasses
import os
from typing import List, Optional

import peft
import torch
from loguru import logger
from peft import (
    get_peft_model,  # type: ignore
    prepare_model_for_kbit_training,  # type: ignore
)
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_utils import _load_state_dict_into_model
from trl.trainer.utils import peft_module_casting_to_bf16

from scarabs.args_factory import ModelArguments
from scarabs.mora.utils.tools import set_color


class ModelFactory:
    def __init__(
        self,
        model_args: ModelArguments,
        model: Optional[PreTrainedModel] = None,
        config: Optional[PretrainedConfig] = None,
        llm_tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        self.model_args = model_args
        self.model = model
        self.config = config
        self.llm_tokenizer = llm_tokenizer

    def handle(self):
        if self.model is None:
            raise ValueError("model is required.")

        self.model = self._weight_init(self.model, self.model_args.model_name_or_path)
        self.model = self._model_setup(self.model)
        return self.model

    def _weight_init(self, model: PreTrainedModel, model_name_or_path=None):
        return model

    def _model_setup(self, model: PreTrainedModel):
        for _, param in model.named_parameters():
            param.requires_grad = True
        self._calulate_parameters(model)
        return model

    def _init_model(self):
        if self.config is None:
            config_kwargs = {
                "cache_dir": self.model_args.cache_dir,
                "revision": self.model_args.model_revision,
                "token": self.model_args.token,
                "trust_remote_code": self.model_args.trust_remote_code,
            }

            if self.model_args.config_name:
                self.config = AutoConfig.from_pretrained(
                    self.model_args.config_name, **config_kwargs
                )
            elif self.model_args.model_name_or_path:
                self.config = AutoConfig.from_pretrained(
                    self.model_args.model_name_or_path, **config_kwargs
                )
            else:
                raise ValueError("model_name_or_path or config_name is required.")

        if self.llm_tokenizer is None:
            tokenizer_kwargs = {
                "cache_dir": self.model_args.cache_dir,
                "use_fast": self.model_args.use_fast_tokenizer,
                "revision": self.model_args.model_revision,
                "token": self.model_args.token,
                "trust_remote_code": self.model_args.trust_remote_code,
            }

            if self.model_args.tokenizer_name:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_args.tokenizer_name, **tokenizer_kwargs
                )
            elif self.model_args.model_name_or_path:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_args.model_name_or_path, **tokenizer_kwargs
                )
            else:
                raise ValueError(
                    "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                    "You can do it from another script, save it, and load it from here, using --tokenizer_name."
                )

        if self.model is None:
            if self.model_args.model_name_or_path:
                torch_dtype = (
                    self.model_args.torch_dtype
                    if self.model_args.torch_dtype in ["auto", None]
                    else getattr(torch, self.model_args.torch_dtype)
                )
                # need to determine the type of model
                architectures = getattr(self.config, "architectures", [])
                if len(architectures) > 0:
                    architectures = architectures[0]
                if "CausalLM" in architectures:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                        config=self.config,
                        cache_dir=self.model_args.cache_dir,
                        revision=self.model_args.model_revision,
                        token=self.model_args.token,
                        trust_remote_code=self.model_args.trust_remote_code,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=self.model_args.low_cpu_mem_usage,
                    )
                elif "MaskedLM" in architectures:
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        self.model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                        config=self.config,
                        cache_dir=self.model_args.cache_dir,
                        revision=self.model_args.model_revision,
                        token=self.model_args.token,
                        trust_remote_code=self.model_args.trust_remote_code,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=self.model_args.low_cpu_mem_usage,
                    )
                else:
                    self.model = AutoModel.from_pretrained(
                        self.model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                        config=self.config,
                        cache_dir=self.model_args.cache_dir,
                        revision=self.model_args.model_revision,
                        token=self.model_args.token,
                        trust_remote_code=self.model_args.trust_remote_code,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=self.model_args.low_cpu_mem_usage,
                    )
            else:
                raise ValueError("model_name_or_path is required.")

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.llm_tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.llm_tokenizer))

    def _load_state_dict(self, model: PreTrainedModel, directory: Optional[str] = None):
        if directory is not None:
            # load safetensors weight
            model_files = self._get_filenames(directory, "safetensors")
            state_dict = {}
            for _file in tqdm(model_files):
                state_dict.update(self._get_safetensors_model_state_dict(_file))

            _load_state_dict_into_model(model, state_dict, "")
            for k in state_dict.keys():
                if "emb" in k:
                    before = state_dict[k][0]
                    if hasattr(model, k):
                        after = getattr(model, k)[0]
                    else:
                        continue
                    self._sanity_check(before[:5], after[:5])
                    break
            del state_dict

        return model

    def _get_filenames(self, directory, suffix=""):
        filenames = []
        files = os.listdir(directory)
        for fi in files:
            tmp_file = os.path.join(directory, fi)
            if os.path.isfile(tmp_file):
                if tmp_file.endswith(suffix):
                    filenames.append(tmp_file)
        return filenames

    def _get_safetensors_model_state_dict(self, model_path):
        """返回 safetensors 模型的 state_dict"""
        state_dict = load_file(model_path)
        return state_dict

    def _sanity_check(self, before: List, after: List):
        logger.info("\n Sanity Check >>>>>>>>>>>>> \n")
        for t, m in zip(before, after):
            logger.info("\n %6f -> %6f \n" % (t, m))
        logger.info("\n <<<<<<<<<<<<< Sanity Check \n")

        assert len(before) == len(
            after
        ), f"length mismatch: {len(before)} vs {len(after)}"

    def _calulate_parameters(self, model):
        """
        Calculate the number of parameters in the model.
        """
        trainable_params = 0
        non_trainable_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                non_trainable_params += param.numel()

        total_params = trainable_params + non_trainable_params
        total = round(total_params / 1e6, 2)
        logger.info(f"Trainable parameters: {trainable_params}")
        logger.info(f"Non-trainable parameters: {non_trainable_params}")
        logger.info(f"Total parameters: {total_params}({total}M)")


class ModelFactoryWithPretrain(ModelFactory):
    def handle(self):
        self._init_model()
        if self.model is None:
            raise ValueError("model is required.")

        self.model = self._model_setup(self.model)
        return self.model

    def _weight_init(self, model: PreTrainedModel, model_name_or_path=None):
        return self._load_state_dict(model, model_name_or_path)

    def _model_setup(self, model: PreTrainedModel):
        for _, param in model.named_parameters():
            param.requires_grad = True
        self._calulate_parameters(model)
        return model


class ModelFactoryWithSFTtrain(ModelFactory):
    def handle(self):
        self._init_model()
        if self.model is None:
            raise ValueError("model is required.")
        if self.model_args.model_name_or_path is not None:
            self._weight_init(self.model, self.model_args.model_name_or_path)
        else:
            logger.info(
                set_color(
                    "model_name_or_path not be set, initialize from scratch", "red"
                )
            )
        self.model = self._model_setup(self.model, self.model_args)
        return self.model

    def _weight_init(self, model: PreTrainedModel, model_name_or_path):
        return self._load_state_dict(model, model_name_or_path)

    def _model_setup(self, model: PreTrainedModel, conf: ModelArguments):
        if conf.peft_config is None:
            for _, param in model.named_parameters():
                param.requires_grad = True
            self._calulate_parameters(model)
        else:
            name = conf.peft_config.pop("name")
            peft_config = getattr(peft, name)(**conf.peft_config)
            if name == "LoraConfig":
                self._set_lora_module_names(model, peft_config)

            self._prepare_peft_model(model, peft_config, conf)
        return model

    def _prepare_peft_model(self, model, peft_config, args):
        """Prepares a model for PEFT training."""
        # Handle quantized models (QLoRA)
        is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(
            model, "is_loaded_in_8bit", False
        )

        is_sharded_qlora = False
        if getattr(model, "is_loaded_in_4bit", False):
            # Check if model is sharded (FSDP/DS-Zero3)
            for _, param in model.named_parameters():
                if param.__class__.__name__ == "Params4bit":
                    is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                    break

        # Prepare model for kbit training if needed
        if is_qlora and not is_sharded_qlora:
            model = self._prepare_model_for_kbit_training(model, args)
            # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
            args = dataclasses.replace(args, model_gradient_checkpointing=False)
        elif args.model_gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Create PEFT model
        if getattr(model, "is_loaded_in_4bit", False) and is_sharded_qlora:
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        else:
            model = get_peft_model(model, peft_config)

        # Handle bf16 casting for 4-bit models
        if (
            args.model_bf16
            and getattr(model, "is_loaded_in_4bit", False)
            and not is_sharded_qlora
        ):
            peft_module_casting_to_bf16(model)

        return model

    def _prepare_model_for_kbit_training(self, model: PreTrainedModel, args):
        """Prepares a quantized model for kbit training."""
        prepare_model_kwargs = {
            "use_gradient_checkpointing": args.model_gradient_checkpointing,
            "gradient_checkpointing_kwargs": args.model_gradient_checkpointing_kwargs
            or {},
        }

        return prepare_model_for_kbit_training(model, **prepare_model_kwargs)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args):
        """Enables gradient checkpointing for the model."""
        model_gradient_checkpointing_kwargs = (
            args.model_gradient_checkpointing_kwargs or {}
        )
        use_reentrant = (
            "use_reentrant" not in model_gradient_checkpointing_kwargs
            or model_gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        return model

    def _set_lora_module_names(self, model, peft_config):
        if peft_config.target_modules is None:
            # get k,q,v,o
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, torch.nn.Linear)
                    and "mlp" not in name
                    and "lm_head" not in name
                ):
                    lora_module_names.add(name.split(".")[-1])
            peft_config.target_modules = list(lora_module_names)
        logger.info(set_color("lora_module_names >>>>>>>>>>>>> \n", "green"))
        for m in peft_config.target_modules:
            logger.info(set_color(f"{m}", "green"))
        logger.info(set_color("<<<<<<<<<<<<< lora_module_names \n", "green"))


class ModelFactoryWithLLMClassification(ModelFactory):
    def handle(self):
        self._init_model()
        if self.model is None:
            raise ValueError("model is required.")
        if self.model_args.model_name_or_path is None:
            raise ValueError("model_name_or_path is required.")
        self.model = self._weight_init(self.model, self.model_args.model_name_or_path)

        self.model = self._model_setup(self.model)
        return self.model

    def _weight_init(self, model: PreTrainedModel, model_name_or_path: str):
        return self._load_state_dict(model, model_name_or_path)

    def _model_setup(self, model: PreTrainedModel):
        for _, param in model.named_parameters():
            param.requires_grad = True
        self._calulate_parameters(model)
        return model


class ModelFactoryWithTabular(ModelFactory):
    def handle(self):
        if self.model is None:
            raise ValueError("model is required.")

        self.model = self._weight_init(self.model, self.model_args.model_name_or_path)
        self.model = self._model_setup(self.model)
        return self.model

    def _weight_init(self, model: PreTrainedModel, model_name_or_path=None):
        return self._load_state_dict(model, model_name_or_path)

    def _model_setup(self, model: PreTrainedModel):
        for _, param in model.named_parameters():
            param.requires_grad = True
        self._calulate_parameters(model)
        return model


# class ModelFactoryWithSFTLoratrain(ModelFactory):
#     def handle(self):
#         self._init_model()
#         if self.model is None:
#             raise ValueError("model is required.")

#         self._weight_init(self.model, self.model_args.model_name_or_path)  # type: ignore
#         self.model = self._model_setup(self.model, self.model_args)  # type: ignore
#         return self.model

#     def _weight_init(self, model: PreTrainedModel, model_name_or_path):
#         return self._load_state_dict(model, model_name_or_path)

#     def _model_setup(self, model: PreTrainedModel, conf: ModelArguments):
#         # For backward compatibility
#         if hasattr(model, "enable_input_require_grads"):
#             model.enable_input_require_grads()
#         else:

#             def make_inputs_require_grad(module, input, output):
#                 output.requires_grad_(True)

#             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
#         if self.model_args.lora_target_modules is not None:
#             lora_module_names = self.model_args.lora_target_modules
#         else:
#             # get k,q,v,o
#             lora_module_names = set()
#             for name, module in model.named_modules():
#                 if (
#                     isinstance(module, torch.nn.Linear)
#                     and "mlp" not in name
#                     and "lm_head" not in name
#                 ):
#                     lora_module_names.add(name.split(".")[-1])
#         logger.info(set_color("\n lora_module_names >>>>>>>>>>>>> \n", "green"))
#         for m in lora_module_names:
#             logger.info(set_color(f"{m}", "green"))
#         logger.info(set_color("\n <<<<<<<<<<<<< lora_module_names \n", "green"))
#         # lora tuning setup
#         peft_config = LoraConfig(
#             task_type=conf.sft_task_type,
#             inference_mode=False,
#             r=conf.lora_r,
#             target_modules=list(lora_module_names),
#             lora_alpha=conf.lora_alpha,
#             lora_dropout=conf.lora_dropout,
#         )
#         peft_llm_model = get_peft_model(model, peft_config)

#         self._calulate_parameters(peft_llm_model)
#         logger.info(
#             f"memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB"
#         )
#         return peft_llm_model
