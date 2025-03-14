# -*- coding: utf-8 -*-
# @Time   : 2024/08/13 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from dataclasses import dataclass, field
from typing import Dict, Optional

from transformers import TrainingArguments
from trl.trainer import DPOConfig


def flatten_dict(nested: Dict, sep: str = "/") -> Dict:
    """Flatten dictionary and concatenate nested keys with separator."""

    def recurse(nest: Dict, prefix: str, into: Dict) -> None:
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Dict):
                recurse(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    recurse(nested, "", flat)
    return flat


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    torch_dtype: str = field(
        default="auto",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code when loading a model."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    sft_task_type: Optional[str] = field(
        default="CAUSAL_LM",
        metadata={
            "help": ("Enum class for the different types of tasks supported by PEFT.")
        },
    )
    peft_config: Optional[Dict] = field(
        default=None,
        metadata={
            "help": ("PEFT config for the different types of tasks supported by PEFT.")
        },
    )
    num_virtual_tokens: int = field(
        default=4,
        metadata={"help": ("num_virtual_tokens (`int`): Number of virtual tokens.")},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "use 8 bit precision for the base model - works only with LoRA"
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "use 4 bit precision for the base model - works only with LoRA"
        },
    )
    model_gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "use gradient checkpointing to save memory at the expense of slower backward pass"
        },
    )
    model_gradient_checkpointing_kwargs: Dict = field(
        default_factory=dict,
        metadata={"help": "kwargs for gradient checkpointing"},
    )
    model_bf16: bool = field(
        default=False,
        metadata={
            "help": "use bfloat16 precision for the base model - works only with LoRA"
        },
    )

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if self.tokenizer_name is None and self.model_name_or_path is not None:
            self.tokenizer_name = self.model_name_or_path


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_cache: str = field(
        default="./encode",
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    extension: Optional[str] = field(
        default=None,
        metadata={"help": "The datasets path (text, csv, json, parquet)"},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    user_feature_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input user feature data file (a text file)."},
    )
    item_feature_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input item feature data file (a text file)."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    template: Optional[str] = field(
        default=None,
        metadata={"help": "An optional template."},
    )

    features: Optional[list[Dict]] = field(
        default=None,
        metadata={"help": "The features of the dataset."},
    )
    labels: Optional[list[str]] = field(
        default=None,
        metadata={"help": "The label names."},
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a training/validation file.")

        if self.extension is None:
            raise ValueError("Need an extension.")


@dataclass
class TrainArguments(TrainingArguments):
    early_stopping_patience: int = field(
        default=20,
        metadata={
            "help": "Execute N steps, stop if all indicators are below the optimal solution"
        },
    )

    early_stopping_threshold: float = field(
        default=1e-7,
        metadata={
            "help": "Execute N steps, Difference, stop if the condition is not met once within the number of judgments"
        },
    )
    train_return_outputs: bool = field(
        default=False,
        metadata={"help": "Enable evaluation and return of training data"},
    )

    top_K: int = field(
        default=10,
        metadata={"help": "top_k for retrieval evaluation, default 10"},
    )

    top_K_candidate_num: int = field(
        default=100,
        metadata={"help": "top_k candidate_num for retrieval evaluation, default 100"},
    )


@dataclass
class DPOTrainArguments(TrainArguments, DPOConfig):
    pass
