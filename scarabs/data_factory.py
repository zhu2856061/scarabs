# -*- coding: utf-8 -*-
# @Time   : 2024/07/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os
import random
from itertools import chain
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from accelerate import PartialState
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    load_from_disk,
)
from loguru import logger
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
from trl.trainer.utils import pad

from scarabs.args_factory import DataArguments
from scarabs.mora.utils.feature_utils import Feature2Transformer
from scarabs.mora.utils.tools import set_color
from scarabs.template_factory import template_dict


class DataFactory:
    def __init__(
        self,
        data_args: DataArguments,
        training_args: TrainingArguments,
        tokenizer: Optional[
            PreTrainedTokenizer | PreTrainedTokenizerFast | None
        ] = None,
    ):
        self.IGNORE_INDEX = -100
        self.data_args = data_args
        self.training_args = training_args
        if data_args is not None:
            self.dataset_cache = data_args.dataset_cache
            os.makedirs(data_args.dataset_cache, exist_ok=True)

        self.tokenizer = None
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if self.tokenizer is not None:
            try:
                self.pad_id = self.tokenizer.pad_token_id
                if self.pad_id is None:
                    raise
            except Exception:
                logger.warning(
                    f"The pad_token_id is not set. We set it to {self.tokenizer.eos_token_id}."
                )
                self.pad_id = self.tokenizer.eos_token_id

            if self.pad_id is None:
                raise ValueError("pad_token_id is not set")

            if (
                data_args.max_seq_length is None
                or data_args.max_seq_length > self.tokenizer.model_max_length
            ):
                logger.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                    f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
                )
                data_args.max_seq_length = self.tokenizer.model_max_length
            self.max_seq_length = min(
                data_args.max_seq_length, self.tokenizer.model_max_length
            )

    def get_dataset(self, files=None, template=None):
        if template is None:
            template = self.data_args.template

        if files is not None:
            dataset = self._prepare_dataset(files, template, "test")
            return dataset, None

        train_dataset = None
        validation_dataset = None
        if self.data_args.train_file is not None:
            train_files = self._get_files_abs_path(self.data_args.train_file)
            logger.info(set_color("\n Train Data from Files >>>>>>>>>>>>> \n", "green"))
            for i, m in enumerate(train_files):
                logger.info(set_color("\n %d -> %s \n" % (i, m), "green"))
            logger.info(set_color("\n <<<<<<<<<<<<< Train Data from Files \n", "green"))
            train_dataset = self._prepare_dataset(train_files, template, "train")

        if self.data_args.validation_file is not None:
            val_files = self._get_files_abs_path(self.data_args.validation_file)
            logger.info(set_color("\n Val Data from Files >>>>>>>>>>>>> \n", "green"))
            for i, m in enumerate(val_files):
                logger.info(set_color("\n %d -> %s \n" % (i, m), "green"))
            logger.info(set_color("\n <<<<<<<<<<<<< Val Data from Files \n", "green"))
            validation_dataset = self._prepare_dataset(val_files, template, "val")

        if train_dataset is None and validation_dataset is None:
            raise ValueError("No dataset is provided")
        else:
            return train_dataset, validation_dataset

    def _prepare_dataset(self, files, template, dtype):
        dataset_cache = os.path.join(self.dataset_cache, dtype)

        try:
            if self.data_args.overwrite_cache:
                logger.info(set_color("Overwrite cache", "yellow"))
                raise
            dataset = load_from_disk(dataset_cache, keep_in_memory=False)
            logger.info(
                set_color(
                    f"Finished loading from cache, disk: {dataset_cache}", "yellow"
                )
            )

        except Exception:
            if self.data_args.extension is None:
                raise ValueError("no extension is provided")

            dataset = load_dataset(
                self.data_args.extension,
                data_files=files,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            logger.info(f"dataset basic info: \n {dataset}")

            dataset = self._process(dataset, template, dtype)
            logger.info(f"after _process dataset[0] info: \n {dataset[0]} \n")

            self._sanity_check(dataset, self.tokenizer)

            dataset.save_to_disk(dataset_cache)

        return dataset

    def _sanity_check(self, dataset, tokenizer):
        if (
            dataset is not None
            and dataset[0].get("input_ids") is not None
            and tokenizer is not None
        ):
            tokens = dataset[0]["input_ids"][:10][:-1]
            target = dataset[0]["input_ids"][:10][1:]

            logger.info(set_color("\n Sanity Check >>>>>>>>>>>>> \n", "green"))
            for t, m in zip(tokens, target):
                decoded = tokenizer.decode([t])
                logger.info(
                    set_color("\n %20s: %6d -> %6d \n" % (repr(decoded), t, m), "green")
                )
            logger.info(set_color("\n <<<<<<<<<<<<< Sanity Check \n", "green"))

            assert len(tokens) == len(
                target
            ), f"length mismatch: {len(tokens)} vs {len(target)}"

    def _get_files_abs_path(self, path):
        if isinstance(path, list):
            return [os.path.abspath(p) for p in path]
        elif os.path.isdir(path):
            return [
                os.path.abspath(os.path.join(root, fi))
                for root, dirs, files in os.walk(path)
                for fi in files
            ]
        elif os.path.isfile(path):
            return [os.path.abspath(path)]
        else:
            raise ValueError("Invalid path or unsupported type")

    def _process(
        self,
        dataset: Optional[
            DatasetDict | Dataset | IterableDatasetDict | IterableDataset
        ],
        template: Optional[str] = None,
        dtype=None,
    ):
        raise NotImplementedError

    def data_collator_fn(self, batch_examples):
        raise NotImplementedError


class DataFactoryWithPretrain(DataFactory):
    """your data ,data name is suffix .jsonl or .json

    {"text": "your text"}
    {"text": "your text"}

    """

    def _tokenize_function(self, examples, template):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        if template is not None:
            examples["text"] = [template.format(line) for line in examples["text"]]
        if self.tokenizer is None:
            raise ValueError("tokenizer is initialized")
        # padding = "max_length" if self.data_args.pad_to_max_length else False
        # return self.tokenizer(
        #     examples["text"],
        #     padding=padding,
        #     truncation=True,
        #     max_length=self.max_seq_length,
        # )
        return self.tokenizer(examples["text"])

    def _process(self, dataset: Dataset, template: Optional[str] = None, dtype=None):
        with PartialState().local_main_process_first():
            tokenized_datasets = dataset.map(
                lambda _: self._tokenize_function(_, template),
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=["text"],
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                cache_file_name=f"{self.dataset_cache}/cache/{dtype}.tokenize_function",
            )

        if not self.data_args.line_by_line:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: list(chain(*examples[k])) for k in examples.keys()
                }
                total_length = len(concatenated_examples[list(examples.keys())[0]])

                # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (
                    total_length // self.max_seq_length
                ) * self.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + self.max_seq_length]
                        for i in range(0, total_length, self.max_seq_length)
                    ]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/process#map
            with PartialState().local_main_process_first():
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {self.max_seq_length}",
                    cache_file_name=f"{self.dataset_cache}/cache/{dtype}.group_texts",
                )

        return tokenized_datasets

    def data_collator_fn(self, batch_examples):
        lengths = max(
            [len(x["input_ids"]) for x in batch_examples if x["input_ids"] is not None]
        )
        batch_max_len = min(lengths, self.max_seq_length)

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for example in batch_examples:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]

            # truncate
            input_ids = input_ids[:batch_max_len]
            attention_mask = attention_mask[:batch_max_len]

            # padding
            input_ids_real_len = len(input_ids)
            padding_len = batch_max_len - input_ids_real_len
            input_ids = input_ids + [self.pad_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(input_ids)

        # to tensor
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)

        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

        return inputs


class DataFactoryWithSFT(DataFactory):
    """your data ,data name is suffix .jsonl or .json

    {"system": "your text", "user": "your text", "assistant": "your text"}
    {"system": "your text", "user": "your text", "assistant": "your text"}

    """

    def _tokenize_function(self, examples, template):
        system_txt = ""
        user_txt = ""
        assistant_txt = ""
        if "system" in examples:
            system_txt = examples["system"]
        if "user" in examples:
            user_txt = examples["user"]
        if "assistant" in examples:
            assistant_txt = examples["assistant"]

        # generate new text
        assert len(user_txt) > 0
        assert template is not None
        assert template_dict.get(template) is not None

        if system_txt != "":
            system_txt = template_dict[template].system_format.format(
                content=system_txt
            )
        if user_txt != "":
            user_txt = template_dict[template].user_format.format(content=user_txt)
        if assistant_txt != "":
            assistant_txt = template_dict[template].assistant_format.format(
                content=assistant_txt
            )

        text = ""
        if system_txt != "":
            text += system_txt
        if user_txt != "":
            text += user_txt
        if assistant_txt != "":
            text += assistant_txt

        if self.tokenizer is None:
            raise ValueError("tokenizer is initialized")
        text = self.tokenizer(text)
        text.pop("attention_mask")
        label = self.tokenizer(assistant_txt)

        text["labels"] = [self.IGNORE_INDEX] * (
            len(text["input_ids"]) - len(label["input_ids"])  # type: ignore
        ) + label["input_ids"]

        return text

    def _process(self, dataset: Dataset, template: Optional[str] = None, dtype=None):
        with PartialState().local_main_process_first():
            tokenized_datasets = dataset.map(
                lambda _: self._tokenize_function(_, template),
                batched=False,
                remove_columns=["user", "assistant"],
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                cache_file_name=f"{self.dataset_cache}/cache/{dtype}.tokenize_function",
            )

        return tokenized_datasets

    def data_collator_fn(self, batch_examples):
        lengths = max(
            [len(x["input_ids"]) for x in batch_examples if x["input_ids"] is not None]
        )
        batch_max_len = min(lengths, self.max_seq_length)

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for example in batch_examples:
            input_ids = example["input_ids"]
            attention_mask = [1] * len(example["input_ids"])
            labels = example["labels"]

            # truncate
            input_ids = input_ids[-batch_max_len:]
            attention_mask = attention_mask[-batch_max_len:]
            labels = labels[-batch_max_len:]

            # padding
            input_ids_real_len = len(input_ids)
            padding_len = batch_max_len - input_ids_real_len
            input_ids = [self.pad_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            labels = [self.IGNORE_INDEX] * padding_len + labels

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        # to tensor
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)

        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

        return inputs


class DataFactoryWithDPO(DataFactory):
    """your data ,data name is suffix .jsonl or .json
    {"system": "", "user": "The sky is", "chosen": " blue", "rejected": " green"}
    {"system": "", "user": "The sky is", "chosen": " blue", "rejected": " green"}
    """

    def _tokenize_function(self, examples, template):
        system_txt = ""
        if "system" in examples:
            system_txt = examples["system"]

        user_txt = examples["user"]
        chosen_txt = examples["chosen"]
        rejected_txt = examples["rejected"]

        # generate new text
        assert template is not None
        assert template_dict.get(template) is not None

        if system_txt != "":
            system_txt = template_dict[template].system_format.format(
                content=system_txt
            )
        if user_txt != "":
            user_txt = template_dict[template].user_format.format(content=user_txt)
        if chosen_txt != "":
            chosen_txt = template_dict[template].assistant_format.format(
                content=chosen_txt
            )
        if rejected_txt != "":
            rejected_txt = template_dict[template].assistant_format.format(
                content=rejected_txt
            )

        prompt_text = ""
        if system_txt != "":
            prompt_text += system_txt
        if user_txt != "":
            prompt_text += user_txt

        if self.tokenizer is None:
            raise ValueError("tokenizer is initialized")
        prompt_text = self.tokenizer(prompt_text)["input_ids"]
        rejected_txt = self.tokenizer(rejected_txt)["input_ids"]
        chosen_txt = self.tokenizer(chosen_txt)["input_ids"]

        return {
            "prompt_input_ids": prompt_text,
            "chosen_input_ids": chosen_txt,
            "rejected_input_ids": rejected_txt,
        }

    def _process(self, dataset: Dataset, template: Optional[str] = None, dtype=None):
        with PartialState().local_main_process_first():
            tokenized_datasets = dataset.map(
                lambda _: self._tokenize_function(_, template),
                batched=False,
                remove_columns=["user", "chosen", "rejected"],
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                cache_file_name=f"{self.dataset_cache}/cache/{dtype}.tokenize_function",
            )

        return tokenized_datasets

    def data_collator_fn(self, batch_examples):
        # Convert to tensor
        prompt_input_ids = [
            torch.tensor(example["prompt_input_ids"]) for example in batch_examples
        ]
        prompt_attention_mask = [
            torch.ones_like(input_ids) for input_ids in prompt_input_ids
        ]
        chosen_input_ids = [
            torch.tensor(example["chosen_input_ids"]) for example in batch_examples
        ]
        chosen_attention_mask = [
            torch.ones_like(input_ids) for input_ids in chosen_input_ids
        ]
        rejected_input_ids = [
            torch.tensor(example["rejected_input_ids"]) for example in batch_examples
        ]
        rejected_attention_mask = [
            torch.ones_like(input_ids) for input_ids in rejected_input_ids
        ]
        if "pixel_values" in batch_examples[0]:
            pixel_values = [
                torch.tensor(example["pixel_values"]) for example in batch_examples
            ]
        if "pixel_attention_mask" in batch_examples[0]:
            pixel_attention_mask = [
                torch.tensor(example["pixel_attention_mask"])
                for example in batch_examples
            ]
        if (
            "ref_chosen_logps" in batch_examples[0]
            and "ref_rejected_logps" in batch_examples[0]
        ):
            ref_chosen_logps = torch.tensor(
                [example["ref_chosen_logps"] for example in batch_examples]
            )
            ref_rejected_logps = torch.tensor(
                [example["ref_rejected_logps"] for example in batch_examples]
            )

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(
            prompt_input_ids,
            padding_value=self.pad_id,  # type: ignore
            padding_side="left",
        )
        output["prompt_attention_mask"] = pad(
            prompt_attention_mask, padding_value=0, padding_side="left"
        )
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_id)  # type: ignore
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(
            rejected_input_ids,
            padding_value=self.pad_id,  # type: ignore
        )
        output["rejected_attention_mask"] = pad(
            rejected_attention_mask, padding_value=0
        )
        if "pixel_values" in batch_examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0)
        if "pixel_attention_mask" in batch_examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in batch_examples[0]:
            output["image_sizes"] = torch.tensor(
                [example["image_sizes"] for example in batch_examples]
            )
        if (
            "ref_chosen_logps" in batch_examples[0]
            and "ref_rejected_logps" in batch_examples[0]
        ):
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        return output


class DataFactoryWithLLMClassification(DataFactory):
    """your data ,data name is suffix .jsonl or .json

    {"text": "your text", "label": "class"}
    {"text": "your text", "label": "class"}

    """

    def _tokenize_function(self, examples, template):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        if template is not None:
            examples["text"] = [template.format(line) for line in examples["text"]]
        if self.tokenizer is None:
            raise ValueError("tokenizer is initialized")

        out = self.tokenizer(examples["text"])
        out["labels"] = examples["label"]
        return out

    def _process(self, dataset: Dataset, template: Optional[str] = None, dtype=None):
        with PartialState().local_main_process_first():
            tokenized_datasets = dataset.map(
                lambda _: self._tokenize_function(_, template),
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=["text"],
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                cache_file_name=f"{self.dataset_cache}/cache/{dtype}.tokenize_function",
            )
        return tokenized_datasets

    def data_collator_fn(self, batch_examples):
        lengths = max(
            [len(x["input_ids"]) for x in batch_examples if x["input_ids"] is not None]
        )
        batch_max_len = min(lengths, self.max_seq_length)

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for example in batch_examples:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            labels = example["labels"]

            # truncate
            input_ids = input_ids[:batch_max_len]
            attention_mask = attention_mask[:batch_max_len]

            # padding
            input_ids_real_len = len(input_ids)
            padding_len = batch_max_len - input_ids_real_len
            input_ids = input_ids + [self.pad_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        # to tensor
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)

        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

        return inputs


class DataFactoryWithTabular(DataFactory):
    def create_feature2transformer(self, config, files=None):
        self.FT = Feature2Transformer(
            os.path.join(self.data_args.dataset_cache, "meta")
        )
        self.FT.create_meta(config)

        if self.data_args.extension is None:
            raise ValueError("no extension is provided")
        if files is None:
            files = self._get_files_abs_path(self.data_args.train_file)
            logger.info("\n Meta Data from Files >>>>>>>>>>>>> \n")
            for i, m in enumerate(files):
                logger.info("\n %d -> %s \n" % (i, m))
            logger.info("\n <<<<<<<<<<<<< Meta Data from Files \n")

        dataset = load_dataset(
            self.data_args.extension,
            data_files=files,
            split="train",
            cache_dir=f"{self.dataset_cache}/cache/feature2transformer",
        )
        if not isinstance(dataset, Dataset):
            raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")

        valid_columns = list(self.FT.feature2meta.keys())
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in valid_columns]
        )
        logger.info(f"dataset basic info: \n {dataset}")
        with PartialState().local_main_process_first():
            dataset.map(
                self.FT.build_meta_batch,
                batched=True,
                batch_size=10000,
                # num_proc=self.data_args.preprocessing_num_workers,
                desc="Running FT build_meta_batch on dataset",
                cache_file_name=f"{self.dataset_cache}/cache/meta.build_meta_batch",
            )
        for item in self.FT.feature2meta.items():
            tmp = f"【{item[0]}】 vocab size: {len(item[1].vocab)}"
            logger.info(set_color(tmp, "green"))

        # self.FT.save_meta()

    def load_feature2transformer(self, config):
        try:
            self.FT = Feature2Transformer(
                os.path.join(self.data_args.dataset_cache, "meta")
            )
        except Exception:
            self.FT = Feature2Transformer()
        self.FT.load_meta(config)
        for item in self.FT.feature2meta.items():
            logger.info(
                set_color(f"【{item[0]}】 vocab size: {len(item[1].vocab)}", "green")
            )

    def get_feature2meta(self):
        return self.FT.feature2meta

    def _process(self, dataset: Dataset, template: Optional[str] = None, dtype=None):
        label_names = (
            []
            if self.training_args.label_names is None
            else self.training_args.label_names
        )
        valid_columns = list(self.FT.feature2meta.keys()) + label_names
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in valid_columns]
        )
        logger.info(set_color(f"dataset basic info: \n {dataset}", "yellow"))
        with PartialState().local_main_process_first():
            dataset = dataset.map(
                self.FT.handle,
                batched=False,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Running FT handle on dataset",
                cache_file_name=f"{self.dataset_cache}/cache/{dtype}.FT_handle",
            )
        logger.info(set_color(f"after handle example: \n {dataset[0]}", "yellow"))
        return dataset

    def data_collator_fn(self, batch_examples):
        X = {}
        for example in batch_examples:
            for name, value in example.items():
                if name not in X:
                    X[name] = []
                X[name].append(value)

        label_names = (
            []
            if self.training_args.label_names is None
            else self.training_args.label_names
        )
        for name in X:
            if name in label_names:
                X[name] = torch.tensor(X[name], dtype=torch.long)
            else:
                X[name] = torch.tensor(X[name], dtype=torch.long)

        return X


class DataFactoryWithTabularRecall(DataFactory):
    def create_feature2transformer(self, config):
        self.FT = Feature2Transformer(
            os.path.join(self.data_args.dataset_cache, "meta")
        )
        self.FT.create_meta(config)
        if self.data_args.extension is None:
            raise ValueError("no extension is provided")

        # base
        train_file = self._get_files_abs_path(self.data_args.train_file)
        dataset = load_dataset(
            self.data_args.extension,
            data_files=train_file,
            split="train",
            cache_dir=os.path.join(self.dataset_cache, "cache"),
        )
        if not isinstance(dataset, Dataset):
            raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")
        valid_columns = list(self.FT.feature2meta.keys())
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in valid_columns]
        )
        logger.info(set_color(f"dataset basic info: \n {dataset})", "yellow"))
        dataset.map(
            self.FT.build_meta_batch,
            batched=True,
            batch_size=10000,
            # num_proc=self.data_args.preprocessing_num_workers,
            desc="Running FT build_meta on dataset line_by_line",
            cache_file_name="tmp/tmp.build_meta_batch",
        )

        # user
        if self.data_args.user_feature_file is not None:
            user_feature_file = self._get_files_abs_path(
                self.data_args.user_feature_file
            )
            user_feature_dataset = load_dataset(
                self.data_args.extension,
                data_files=user_feature_file,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(user_feature_dataset, Dataset):
                raise ValueError(
                    f"dataset is not a Dataset, but {type(user_feature_dataset)}"
                )
            user_feature_dataset = user_feature_dataset.remove_columns(
                [
                    col
                    for col in user_feature_dataset.column_names
                    if col not in valid_columns
                ]
            )
            logger.info(
                set_color(f"dataset basic info: \n {user_feature_dataset})", "yellow")
            )
            user_feature_dataset.map(
                self.FT.build_meta_batch,
                batched=True,
                batch_size=10000,
                desc="Running FT build_meta on dataset line_by_line",
                cache_file_name="tmp/tmp.build_meta_batch",
            )

        # item
        if self.data_args.item_feature_file is not None:
            item_feature_file = self._get_files_abs_path(
                self.data_args.item_feature_file
            )
            item_feature_dataset = load_dataset(
                self.data_args.extension,
                data_files=item_feature_file,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(item_feature_dataset, Dataset):
                raise ValueError(
                    f"dataset is not a Dataset, but {type(item_feature_dataset)}"
                )
            item_feature_dataset = item_feature_dataset.remove_columns(
                [
                    col
                    for col in item_feature_dataset.column_names
                    if col not in valid_columns
                ]
            )
            logger.info(
                set_color(f"dataset basic info: \n {item_feature_dataset})", "yellow")
            )
            item_feature_dataset.map(
                self.FT.build_meta_batch,
                batched=True,
                batch_size=10000,
                desc="Running FT build_meta on dataset line_by_line",
                cache_file_name="tmp/tmp.build_meta_batch",
            )

        for item in self.FT.feature2meta.items():
            logger.info(
                set_color(f"【{item[0]}】 vocab size: {len(item[1].vocab)}", "green")
            )

    def load_feature2transformer(self, config):
        try:
            self.FT = Feature2Transformer(
                os.path.join(self.data_args.dataset_cache, "meta")
            )
        except Exception:
            self.FT = Feature2Transformer()

        self.FT.load_meta(config)
        for item in self.FT.feature2meta.items():
            logger.info(
                set_color(f"【{item[0]}】 vocab size: {len(item[1].vocab)}", "green")
            )

        self._load_user_feature2transformer()
        self._load_item_feature2transformer()

    def _load_user_feature2transformer(self):
        self.user_feature = {}
        if self.data_args.user_feature_file is not None:
            user_feature_file = self._get_files_abs_path(
                self.data_args.user_feature_file
            )

            if self.data_args.extension is None:
                raise ValueError("no extension is provided")
            user_feature_dataset = load_dataset(
                self.data_args.extension,
                data_files=user_feature_file,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )

            def insert_feature(_):
                self.user_feature[_["uid"]] = _

            user_feature_dataset.map(insert_feature)

    def _load_item_feature2transformer(self):
        self.item_feature = {}
        if self.data_args.item_feature_file is not None:
            item_feature_file = self._get_files_abs_path(
                self.data_args.item_feature_file
            )

            if self.data_args.extension is None:
                raise ValueError("no extension is provided")
            item_feature_dataset = load_dataset(
                self.data_args.extension,
                data_files=item_feature_file,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )

            self.item_feature = {}

            def insert_feature(_):
                self.item_feature[_["uid"]] = _

            item_feature_dataset.map(insert_feature)

    def get_feature2meta(self):
        return self.FT.feature2meta

    def get_train_dataset(self, files=None, template=None):
        if files is not None:
            dataset = self._prepare_train_dataset(files, template, "train")
            return dataset

        dataset = None
        if self.data_args.train_file is not None:
            train_file = self._get_files_abs_path(self.data_args.train_file)
            dataset = self._prepare_train_dataset(train_file, template, "train")

        return dataset

    def _prepare_train_dataset(self, files=None, template=None, dtype=None):
        dataset_cache = self.data_args.dataset_cache
        if dtype is not None:
            dataset_cache = os.path.join(dataset_cache, dtype)
        try:
            if self.data_args.overwrite_cache:
                logger.info(set_color("Overwrite cache", "red"))
                raise
            dataset = load_from_disk(dataset_cache, keep_in_memory=False)
            logger.info(set_color("Finished loading from cache", "green"))
            if dataset[0].get("input_ids") is not None:  # type: ignore
                self._sanity_check(
                    dataset[0]["input_ids"][:10][:-1], dataset[0]["input_ids"][:10][1:]
                )
        except Exception:
            if self.data_args.extension is None:
                raise ValueError("no extension is provided")
            dataset = load_dataset(
                self.data_args.extension,
                data_files=files,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(dataset, Dataset):
                raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")
            dataset = self._train_process(dataset, template)
            logger.info(
                set_color(
                    f"after format_torch dataset[0] info: \n {dataset[0]} \n", "yellow"
                )
            )
            if dataset[0].get("input_ids") is not None:
                self._sanity_check(
                    dataset[0]["input_ids"][:10][:-1], dataset[0]["input_ids"][:10][1:]
                )
            dataset.save_to_disk(dataset_cache)

        if isinstance(dataset, DatasetDict):
            raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")
        return dataset

    def _train_process(self, dataset: Dataset, template: Optional[str] = None):
        label_names = (
            []
            if self.training_args.label_names is None
            else self.training_args.label_names
        )
        valid_columns = list(self.FT.feature2meta.keys()) + label_names
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in valid_columns]
        )
        logger.info(f"dataset basic info: \n {dataset}")

        # joint feature
        def joint_feature(example):
            uid = self.user_feature.get(example["uid"], {"uid": example["uid"]})
            iid = self.item_feature.get(example["iid"], {"iid": example["iid"]})

            iid.update(uid)
            iid = self.FT.handle(iid)
            return iid

        dataset = dataset.map(
            joint_feature,
            batched=False,
            num_proc=self.data_args.preprocessing_num_workers,
            desc="Running joint feature on dataset line_by_line",
            cache_file_name="tmp/train.joint_feature",
        )

        logger.info(set_color(f"after handle example:  \n {dataset[0]}", "yellow"))
        return dataset

    def get_eval_dataset(self, files=None, template=None):
        if files is not None:
            dataset = self._prepare_eval_dataset(files, template, "val")
            return dataset

        dataset = None
        if self.data_args.validation_file is not None:
            validation_file = self._get_files_abs_path(self.data_args.validation_file)
            dataset = self._prepare_eval_dataset(validation_file, template, "val")

        return dataset

    def _prepare_eval_dataset(self, files=None, template=None, dtype=None):
        dataset_cache = self.data_args.dataset_cache
        if dtype is not None:
            dataset_cache = os.path.join(dataset_cache, dtype)
        try:
            if self.data_args.overwrite_cache:
                logger.info(set_color("Overwrite cache", "red"))
                raise
            dataset = load_from_disk(dataset_cache, keep_in_memory=False)
            logger.info(set_color("Finished loading from cache", "green"))
            if dataset[0].get("input_ids") is not None:  # type: ignore
                self._sanity_check(
                    dataset[0]["input_ids"][:10][:-1], dataset[0]["input_ids"][:10][1:]
                )
        except Exception:
            if self.data_args.extension is None:
                raise ValueError("no extension is provided")
            dataset = load_dataset(
                self.data_args.extension,
                data_files=files,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(dataset, Dataset):
                raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")
            dataset = self._eval_process(dataset, template)
            logger.info(
                set_color(
                    f"after format_torch dataset[0] info: \n {dataset[0]} \n", "yellow"
                )
            )
            if dataset[0].get("input_ids") is not None:
                self._sanity_check(
                    dataset[0]["input_ids"][:10][:-1], dataset[0]["input_ids"][:10][1:]
                )
            dataset.save_to_disk(dataset_cache)

        if isinstance(dataset, DatasetDict):
            raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")
        return dataset

    def _eval_process(self, dataset: Dataset, template: Optional[str] = None):
        """
        dataset:
        {"uid": "xxx", "iid": "xxx,xxx,xxx"}
        其中 iid 为 1个正例， neg_iid 为 N个负例
        评估的核心 为 正例 在 N+1 个例子中的 @K rank
        """

        # joint feature
        def joint_feature(example):
            examples = []
            iid_list = example["iid"].split("#")

            uid = self.user_feature.get(example["uid"], {"uid": example["uid"]})

            examples = []
            for neg_iid in iid_list:
                neg_iid = self.item_feature.get(neg_iid, {"iid": neg_iid})
                neg_iid.update(uid)

                neg_iid = self.FT.handle(neg_iid)
                examples.append(neg_iid)

            return {"example": examples}

        dataset = dataset.map(
            joint_feature,
            batched=False,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=["uid", "iid"],
            desc="Running joint feature on dataset line_by_line",
            cache_file_name="tmp/eval.joint_feature",
        )
        logger.info(set_color(f"after handle example:  \n {dataset[0]}", "yellow"))
        return dataset

    def data_collator_fn(self, batch_examples):
        X = {}
        for example in batch_examples:
            if "example" in example:  # eval dataset
                example = example["example"]
                for ex in example:
                    for name, value in ex.items():
                        if name not in X:
                            X[name] = []
                        X[name].append(value)

            else:  # train dataset
                for name, value in example.items():
                    if name not in X:
                        X[name] = []
                    X[name].append(value)

        label_names = (
            []
            if self.training_args.label_names is None
            else self.training_args.label_names
        )
        for name in X:
            if name in label_names:
                X[name] = torch.tensor(X[name], dtype=torch.long)
            else:
                X[name] = torch.tensor(X[name], dtype=torch.long)

        return X


class DataFactoryWithTabularRecall2(DataFactory):
    def create_feature2transformer(self, config):
        self.FT = Feature2Transformer(
            os.path.join(self.data_args.dataset_cache, "meta")
        )
        self.FT.create_meta(config)

        # base - [user, item (list)]
        train_file = self._get_files_abs_path(self.data_args.train_file)
        dataset = load_dataset(
            "text",
            data_files=train_file,
            split="train",
            cache_dir=os.path.join(self.dataset_cache, "cache"),
        )

        if not isinstance(dataset, Dataset):
            raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")
        logger.info(set_color(f"dataset basic info: \n {dataset})", "yellow"))
        dataset.map(
            self._batch_text_to_user_item,
            batched=True,
            batch_size=10000,
            desc="Running FT build_meta on dataset line_by_line",
            cache_file_name="tmp/tmp.build_meta_batch",
        )

        # user
        if self.data_args.user_feature_file is not None:
            user_feature_file = self._get_files_abs_path(
                self.data_args.user_feature_file
            )
            user_feature_dataset = load_dataset(
                "json",
                data_files=user_feature_file,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(user_feature_dataset, Dataset):
                raise ValueError(
                    f"dataset is not a Dataset, but {type(user_feature_dataset)}"
                )
            user_feature_dataset = user_feature_dataset.remove_columns(
                [col for col in user_feature_dataset.column_names]
            )
            logger.info(
                set_color(f"dataset basic info: \n {user_feature_dataset})", "yellow")
            )
            user_feature_dataset.map(
                self.FT.build_meta_batch,
                batched=True,
                batch_size=10000,
                desc="Running FT build_meta on dataset line_by_line",
                cache_file_name="tmp/tmp.build_meta_batch",
            )

        # item
        if self.data_args.item_feature_file is not None:
            item_feature_file = self._get_files_abs_path(
                self.data_args.item_feature_file
            )
            item_feature_dataset = load_dataset(
                "json",
                data_files=item_feature_file,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(item_feature_dataset, Dataset):
                raise ValueError(
                    f"dataset is not a Dataset, but {type(item_feature_dataset)}"
                )
            item_feature_dataset = item_feature_dataset.remove_columns(
                [col for col in item_feature_dataset.column_names]
            )
            logger.info(
                set_color(f"dataset basic info: \n {item_feature_dataset})", "yellow")
            )
            item_feature_dataset.map(
                self.FT.build_meta_batch,
                batched=True,
                batch_size=10000,
                desc="Running FT build_meta on dataset line_by_line",
                cache_file_name="tmp/tmp.build_meta_batch",
            )

        for item in self.FT.feature2meta.items():
            logger.info(
                set_color(f"【{item[0]}】 vocab size: {len(item[1].vocab)}", "green")
            )

    def _batch_text_to_user_item(self, lines):
        tmps = {"uid": [], "iid": []}
        for line in lines["text"]:
            items = line.strip().split(" ")
            user = items[0]
            items = items[1:]
            tmps["uid"].append(user)
            tmps["iid"].append(items)

        self.FT.build_meta_batch(tmps)

    def load_feature2transformer(self, config):
        try:
            self.FT = Feature2Transformer(
                os.path.join(self.data_args.dataset_cache, "meta")
            )
        except Exception:
            self.FT = Feature2Transformer()

        self.FT.load_meta(config)
        for item in self.FT.feature2meta.items():
            logger.info(
                set_color(f"【{item[0]}】 vocab size: {len(item[1].vocab)}", "green")
            )

        self.n_uids = len(self.FT.feature2meta["uid"].vocab)
        self.n_iids = len(self.FT.feature2meta["iid"].vocab)

        self._load_user_feature2transformer()
        self._load_item_feature2transformer()

    def _load_user_feature2transformer(self):
        self.user_feature = {}
        if self.data_args.user_feature_file is not None:
            _file = self._get_files_abs_path(self.data_args.user_feature_file)

            dataset = load_dataset(
                "json",
                data_files=_file,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(dataset, Dataset):
                raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")

            valid_columns = list(self.FT.feature2meta.keys())

            user_columns = [col for col in dataset.column_names if col in valid_columns]

            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in valid_columns]
            )

            def trans_and_insert_feature(_):
                tmp = self.FT.handle(_)
                tmp = {k: v for k, v in tmp.items() if k in user_columns}
                self.user_feature[tmp["uid"][0]] = tmp

            dataset.map(trans_and_insert_feature)
        else:
            for i in range(self.n_uids):
                self.user_feature[i] = {"uid": [i]}

    def _load_item_feature2transformer(self):
        self.item_feature = {}
        if self.data_args.item_feature_file is not None:
            _file = self._get_files_abs_path(self.data_args.item_feature_file)

            dataset = load_dataset(
                "json",
                data_files=_file,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(dataset, Dataset):
                raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")

            valid_columns = list(self.FT.feature2meta.keys())

            item_columns = [col for col in dataset.column_names if col in valid_columns]

            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in valid_columns]
            )

            def trans_and_insert_feature(_):
                tmp = self.FT.handle(_)
                tmp = {k: v for k, v in tmp.items() if k in item_columns}
                self.item_feature[tmp["iid"][0]] = tmp

            dataset.map(trans_and_insert_feature)
        else:
            for i in range(self.n_iids):
                self.item_feature[i] = {"iid": [i]}

    def get_feature2meta(self):
        return self.FT.feature2meta

    def get_train_dataset(self, files=None, template=None):
        if files is not None:
            dataset = self._prepare_dataset(files, template, "train")
            return dataset

        dataset = None
        if self.data_args.train_file is not None:
            train_file = self._get_files_abs_path(self.data_args.train_file)
            dataset = self._prepare_dataset(train_file, template, "train")

        # 极为重要， 训练的item，本质上是用户已经点击过的物品，故在验证中是不能随机出来的
        self.train_data_dict = {_["uid"]: _["iid"] for _ in dataset}  # type: ignore
        logger.info(
            set_color(
                f"train_data_dict basic info: \n {self.train_data_dict[1]})", "yellow"
            )
        )

        return dataset

    def get_eval_dataset(self, files=None, template=None):
        if files is not None:
            dataset = self._prepare_dataset(files, template, "val")
            return dataset

        dataset = None
        if self.data_args.validation_file is not None:
            validation_file = self._get_files_abs_path(self.data_args.validation_file)
            dataset = self._prepare_dataset(validation_file, template, "val")

        return dataset

    def _prepare_dataset(self, files=None, template=None, dtype=None):
        dataset_cache = self.data_args.dataset_cache
        if dtype is not None:
            dataset_cache = os.path.join(dataset_cache, dtype)
        try:
            if self.data_args.overwrite_cache:
                logger.info(set_color("Overwrite cache", "red"))
                raise
            dataset = load_from_disk(dataset_cache, keep_in_memory=False)
            logger.info(set_color("Finished loading from cache", "green"))
        except Exception:
            dataset = load_dataset(
                "text",
                data_files=files,
                split="train",
                cache_dir=os.path.join(self.dataset_cache, "cache"),
            )
            if not isinstance(dataset, Dataset):
                raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")
            dataset = self._process(dataset, template, dtype)
            dataset.save_to_disk(dataset_cache)

        logger.info(
            set_color(
                f"after format_torch dataset[0] info: \n {dataset[0]} \n", "yellow"
            )
        )
        if isinstance(dataset, DatasetDict):
            raise ValueError(f"dataset is not a Dataset, but {type(dataset)}")
        return dataset

    def _process(self, dataset: Dataset, template: Optional[str] = None, dtype=None):
        """dataset: line by line
        dataset: uid_xxx iid_xxx iid_xxx iid_xxx
        user_items_list
        训练的过程中随机构建负样本
        """

        def _batch_text_to_user_item(lines):
            tmps = {"uid": [], "iid": [], "dtype": []}

            for line in lines["text"]:
                items = line.strip().split(" ")
                user = items[0]
                items = items[1:]
                user = self.FT.feature2meta["uid"].vocab.get("uid_" + user, 0)
                items = [
                    self.FT.feature2meta["iid"].vocab.get("iid_" + item, 0)
                    for item in items
                ]
                tmps["uid"].append(user)
                tmps["iid"].append(items)
                if dtype == "val":
                    tmps["dtype"].append("val")
                else:
                    tmps["dtype"].append("train")

            return tmps

        dataset = dataset.map(
            _batch_text_to_user_item,
            batched=True,
            remove_columns=["text"],
            batch_size=10000,
            desc="Running train data on dataset line_by_line",
            cache_file_name=f"tmp/{dtype}.dataset",
        )

        logger.info(
            set_color(f"{dtype} after handle example:  \n {dataset[0]}", "yellow")
        )
        return dataset

    def data_collator_fn(self, batch_examples):
        def stark_to_tensor(example):
            X = {}
            for line in example:
                for k, v in line.items():
                    if k not in X:
                        X[k] = []
                    X[k].append(v)

            for name in X:
                X[name] = torch.tensor(X[name], dtype=torch.long)
            return X

        def train_pos_neg_item(iid, n_iids):
            # 随机一个正例
            pos_iid = random.sample(iid, 1)[0]
            # 随机一个负例
            while True:
                neg_iid = random.randint(0, n_iids)
                if neg_iid not in iid:
                    break
            return pos_iid, neg_iid

        uids = []
        pos_iids = []
        neg_iids = []

        hist_iids = []
        curr_iids = []

        for example in batch_examples:
            uid = example["uid"]
            iid = example["iid"]
            dtype = example["dtype"]
            if dtype == "train":
                pos_iid, neg_iid = train_pos_neg_item(iid, self.n_iids)
                uid = self.user_feature.get(uid, self.user_feature[0])
                pos_iid = self.item_feature.get(pos_iid, self.item_feature[0])
                neg_iid = self.item_feature.get(neg_iid, self.item_feature[0])

                uids.append(uid)
                pos_iids.append(pos_iid)
                neg_iids.append(neg_iid)

            elif dtype == "val":
                hist_hist = self.train_data_dict.get(uid, [])
                uid = self.user_feature.get(uid, self.user_feature[0])

                uids.append(uid)
                hist_iids.append(hist_hist)
                curr_iids.append(iid)

        if len(pos_iids) > 0:
            uids = stark_to_tensor(uids)
            pos_iids = stark_to_tensor(pos_iids)
            neg_iids = stark_to_tensor(neg_iids)
            return {"uids": uids, "pos_iids": pos_iids, "neg_iids": neg_iids}
        else:
            uids = stark_to_tensor(uids)
            return {"uids": uids, "hist_iids": hist_iids, "curr_iids": curr_iids}
