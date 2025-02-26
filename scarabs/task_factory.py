# -*- coding: utf-8 -*-
# @Time   : 2024/07/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys
from typing import Optional, Tuple, Type, Union  # noqa: F401

import datasets
import evaluate
import numpy as np
import torch
import transformers
from loguru import logger
from tqdm import tqdm
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from scarabs.args_factory import DataArguments, ModelArguments, TrainArguments
from scarabs.data_factory import (
    DataFactoryWithLLMClassification,
    DataFactoryWithPretrain,
    DataFactoryWithSFT,
    DataFactoryWithTabular,
    DataFactoryWithTabularRecall,
    DataFactoryWithTabularRecall2,
)
from scarabs.model_factory import (
    ModelFactoryWithContinuePretrain,
    ModelFactoryWithLLMClassification,
    ModelFactoryWithPretrain,
    ModelFactoryWithSFTFulltrain,
    ModelFactoryWithSFTLoratrain,
    ModelFactoryWithTabular,
)
from scarabs.train_factory import (
    EarlyStoppingByEvalDataCallback,
    EarlyStoppingByTrainDataCallback,
    PrettyTablePrinterCallback,
    TrainerFactoryWithLLMClassification,
    TrainerFactoryWithPretrain,
    TrainerFactoryWithTabular,
    TrainerFactoryWithTabularRecall,
    TrainerFactoryWithTabularRecall2,
)

tqdm.pandas()


# task factory
class TaskFactory:
    def __init__(self):
        self.TASK = "Abstract TaskFactory Class"

    def _logging_summary(self, training_args: TrainArguments):
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
        )

        # Set the verbosity to info of the Transformers logger (on main process only):
        logger.info(f"Training/evaluation parameters {training_args}")

    def _load_last_checkpoint(self, training_args: TrainArguments):
        # Detecting last checkpoint.
        self.last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            self.last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if (
                self.last_checkpoint is None
                and len(os.listdir(training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                self.last_checkpoint is not None
                and training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {self.last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

    def _seed(self, training_args: TrainArguments):
        # Set seed before initializing model.
        set_seed(training_args.seed)
        random.seed(training_args.seed)
        os.environ["PYTHONHASHSEED"] = str(training_args.seed)
        np.random.seed(training_args.seed)
        torch.manual_seed(training_args.seed)
        torch.cuda.manual_seed(training_args.seed)
        torch.backends.cudnn.deterministic = True

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        config: Optional[PretrainedConfig] = None,
        llm_tokenizer: Optional[PreTrainedTokenizer] = None,
        ds_train: Optional[datasets.Dataset] = None,
        ds_eval: Optional[datasets.Dataset] = None,
    ):
        raise NotImplementedError


# Pre train factory
class TaskFactoryWithPreTrain(TaskFactory):
    def __init__(self, metrics=["metrics/accuracy"]):
        self.TASK = "TaskFactory PreTrain"

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel],
        config: Optional[PretrainedConfig] = None,
        llm_tokenizer: Optional[PreTrainedTokenizer] = None,
        ds_train: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
        ds_eval: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
    ):
        if model is None:
            raise ValueError("PreTrain need define model")
        # model factory
        modelFactory = ModelFactoryWithPretrain(
            model_args=model_args,
            model=model,
            config=config,
            llm_tokenizer=llm_tokenizer,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # data factory
        dataFactory = DataFactoryWithPretrain(
            data_args=data_args,
            training_args=training_args,
            tokenizer=modelFactory.llm_tokenizer,
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()

        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        #  pretrain no compute metrics

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        trainer = TrainerFactoryWithPretrain(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            data_collator=collator_fn,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )
        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate()

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(
        self,
        model=None,
        model_name_or_path=None,
        modelFunc=None,
        tokenizer=None,
        tokenizer_name_or_path=None,
    ):
        from transformers import AutoTokenizer

        # load model
        self.model = None
        if model is not None:
            self.model = model
        else:
            if model_name_or_path is not None and modelFunc is not None:
                config = PretrainedConfig.from_pretrained(model_name_or_path)
                self.model = modelFunc(config)

        if self.model is None:
            raise ValueError("No model found")

        # load tokenizer
        self.tokenizer = None
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if tokenizer_name_or_path is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        if self.tokenizer is None:
            raise ValueError("No tokenizer found")

        # load weight
        modelFactory = ModelFactoryWithPretrain(
            model_args=None,  # type: ignore
            model=self.model,
            config=config,
            llm_tokenizer=self.tokenizer,  # type: ignore
        )
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X, max_tokens=128):
        if self.model is None:
            raise ValueError("No model found")
        if self.tokenizer is None:
            raise ValueError("No tokenizer found")
        with torch.no_grad():
            i = 0
            res = []
            tokens = [-1]
            X = self.tokenizer(X)
            if X.get("input_ids") is None or X.get("attention_mask") is None:
                return "input X is err"

            while i < max_tokens and tokens[0] != self.tokenizer.pad_token_id:
                inputs = {}
                for name in X:
                    if name not in ["input_ids", "attention_mask"]:
                        continue
                    inputs[name] = torch.tensor([X[name]], dtype=torch.long).to(
                        self.device
                    )

                output = self.model(**inputs)
                logits = output.logits
                tokens = torch.argmax(logits, dim=-1)
                tokens = tokens[0].tolist()[:-2:-1]
                X["input_ids"] = X["input_ids"] + tokens  # type: ignore
                X["attention_mask"] = X["attention_mask"] + [1]  # type: ignore
                res += tokens
                i += 1

            answer = self.tokenizer.decode(res)
            return answer


# LLM Classification
class TaskFactoryWithLLMClassification(TaskFactory):
    def __init__(
        self,
        metrics=[
            "metrics/accuracy",
            "metrics/precision",
            "metrics/recall",
            "metrics/f1",
        ],
    ):
        self.TASK = "TaskFactory LLM text Classification"
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self._metric = evaluate.combine([os.path.join(current_dir, _) for _ in metrics])

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        config: Optional[PretrainedConfig] = None,
        llm_tokenizer: Optional[PreTrainedTokenizer] = None,
        ds_train: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
        ds_eval: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
    ):
        # model factory
        modelFactory = ModelFactoryWithLLMClassification(
            model_args=model_args,
            model=model,
            config=config,
            llm_tokenizer=llm_tokenizer,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # data factory
        dataFactory = DataFactoryWithLLMClassification(
            data_args=data_args,
            training_args=training_args,
            tokenizer=modelFactory.llm_tokenizer,
        )

        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()

        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        # compute metrics
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            logits = logits.argmax(axis=-1)
            res = self._metric.compute(predictions=logits, references=labels)
            return res

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        trainer = TrainerFactoryWithLLMClassification(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=collator_fn,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate()

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(
        self, tokenizer_name_or_path, model_name_or_path, modelFunc
    ):
        from transformers import AutoTokenizer

        config = PretrainedConfig.from_pretrained(model_name_or_path)
        # load model
        self.model = modelFunc(config)
        llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        modelFactory = ModelFactoryWithLLMClassification(
            model_args=None,  # type: ignore
            model=self.model,
            config=config,
            llm_tokenizer=llm_tokenizer,  # type: ignore
        )
        self.model = modelFactory._weight_init(self.model, model_name_or_path)
        self.tokenizer = llm_tokenizer
        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X):
        with torch.no_grad():
            X = self.tokenizer(X)
            for name in X:
                X[name] = torch.tensor([X[name]], dtype=torch.long).to(self.device)
            return self.model(**X)


# Tabular Classification
class TaskFactoryWithTabularClassification(TaskFactory):
    def __init__(
        self,
        metrics=[
            "metrics/accuracy",
            "metrics/precision",
            "metrics/recall",
            "metrics/f1",
        ],
    ):
        self.TASK = "TaskFactory Tabular Classification"
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self._metric = evaluate.combine([os.path.join(current_dir, _) for _ in metrics])

    def create_feature2transformer_and_config(
        self,
        data_args: DataArguments,
        training_args: TrainArguments,
        config: PretrainedConfig,
    ):
        # data factory
        dataFactory = DataFactoryWithTabular(
            data_args=data_args,
            training_args=training_args,
        )

        dataFactory.create_feature2transformer(config.features)

        obj_dict = {}
        for name, fea in dataFactory.get_feature2meta().items():
            obj_dict[name] = fea.__dict__

        config.features = obj_dict
        config.save_pretrained(os.path.join(data_args.dataset_cache, "meta"))

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        modelFunc: Optional[Type[PreTrainedModel]] = None,
        config: Optional[PretrainedConfig] = None,
        ds_train: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
        ds_eval: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
    ):
        if config is None:
            raise ValueError("No config found")

        # data factory
        dataFactory = DataFactoryWithTabular(
            data_args=data_args,
            training_args=training_args,
        )
        dataFactory.load_feature2transformer(config)
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model factory
        if model is None and modelFunc is not None:
            model = modelFunc(config)
        if model is None:
            raise ValueError("No model found")

        modelFactory = ModelFactoryWithTabular(
            model_args=model_args,
            model=model,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        # compute metrics
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            logits = logits.argmax(axis=-1)
            res = self._metric.compute(predictions=logits, references=labels)
            return res

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        trainer = TrainerFactoryWithTabular(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=collator_fn,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(self, model_name_or_path, modelFunc):
        # load data processing
        dataFactory = DataFactoryWithTabular(data_args=None, training_args=None)  # type: ignore
        config = PretrainedConfig.from_pretrained(model_name_or_path)
        dataFactory.load_feature2transformer(config)
        self.feature2trans = dataFactory.FT

        # load model
        self.model = modelFunc(config)
        modelFactory = ModelFactoryWithTabular(model_args=None, model=None)  # type: ignore
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X):
        with torch.no_grad():
            X = self.feature2trans.handle(X)
            for name in X:
                X[name] = torch.tensor([X[name]], dtype=torch.long).to(self.device)  # type: ignore
            return self.model(**X)


# Tabular ctr
class TaskFactoryWithTabularCtr(TaskFactory):
    def __init__(self, metrics=["metrics/roc_auc"]):
        self.TASK = "TaskFactory Tabular Ctr"
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self._metric = evaluate.combine([os.path.join(current_dir, _) for _ in metrics])

    def create_feature2transformer_and_config(
        self,
        data_args: DataArguments,
        training_args: TrainArguments,
        config: PretrainedConfig,
    ):
        # data factory
        dataFactory = DataFactoryWithTabular(
            data_args=data_args,
            training_args=training_args,
        )

        dataFactory.create_feature2transformer(config.features)

        obj_dict = {}
        for name, fea in dataFactory.get_feature2meta().items():
            obj_dict[name] = fea.__dict__

        config.features = obj_dict
        config.save_pretrained(os.path.join(data_args.dataset_cache, "meta"))

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        modelFunc: Optional[Type[PreTrainedModel]] = None,
        config: Optional[PretrainedConfig] = None,
        ds_train: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
        ds_eval: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
    ):
        if config is None:
            raise ValueError("No config found")

        # data factory
        dataFactory = DataFactoryWithTabular(
            data_args=data_args,
            training_args=training_args,
        )
        dataFactory.load_feature2transformer(config)
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model factory
        if model is None and modelFunc is not None:
            model = modelFunc(config)
        if model is None:
            raise ValueError("No model found")

        modelFactory = ModelFactoryWithTabular(
            model_args=model_args,
            model=model,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        # compute metrics
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            res = self._metric.compute(prediction_scores=logits, references=labels)
            return res

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        trainer = TrainerFactoryWithTabular(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=collator_fn,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(self, model_name_or_path, modelFunc):
        # load data processing
        dataFactory = DataFactoryWithTabular(data_args=None, training_args=None)  # type: ignore
        config = PretrainedConfig.from_pretrained(model_name_or_path)
        dataFactory.load_feature2transformer(config)
        self.feature2trans = dataFactory.FT

        # load model
        self.model = modelFunc(config)
        modelFactory = ModelFactoryWithTabular(model_args=None, model=None)  # type: ignore
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X):
        with torch.no_grad():
            X = self.feature2trans.handle(X)
            for name in X:
                X[name] = torch.tensor([X[name]], dtype=torch.long).to(self.device)  # type: ignore
            return self.model(**X)

    def batch_inference(self, batch_X):
        for _ in range(len(batch_X)):
            batch_X[_] = self.feature2trans.handle(batch_X[_])

        X = {}
        for _ in range(len(batch_X)):
            for name in batch_X[_]:
                if name not in X:
                    X[name] = []
                X[name].append(batch_X[_][name])

        with torch.no_grad():
            for name in X:
                X[name] = torch.tensor(X[name], dtype=torch.long).to(self.device)  # type: ignore
            return self.model(**X)


# Tabular Recall
class TaskFactoryWithTabularRecall(TaskFactory):
    def __init__(self, metrics=["metrics/roc_auc", "metrics/topk"]):
        self.TASK = "TaskFactory Tabular recall"
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self._train_metric = evaluate.load(os.path.join(current_dir, metrics[0]))
        self._eval_metric = evaluate.load(os.path.join(current_dir, metrics[1]))

    def create_feature2transformer_and_config(
        self,
        data_args: DataArguments,
        training_args: TrainArguments,
        config: PretrainedConfig,
    ):
        # data factory
        dataFactory = DataFactoryWithTabularRecall(
            data_args=data_args,
            training_args=training_args,
        )

        dataFactory.create_feature2transformer(config.features)

        obj_dict = {}
        for name, fea in dataFactory.get_feature2meta().items():
            obj_dict[name] = fea.__dict__

        config.features = obj_dict
        config.save_pretrained(os.path.join(data_args.dataset_cache, "meta"))

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        modelFunc: Optional[Type[PreTrainedModel]] = None,
        config: Optional[PretrainedConfig] = None,
        ds_train: Optional[datasets.Dataset] = None,
        ds_eval: Optional[datasets.Dataset] = None,
    ):
        if config is None:
            raise ValueError("No config found")

        # data factory
        dataFactory = DataFactoryWithTabularRecall(
            data_args=data_args,
            training_args=training_args,
        )
        dataFactory.load_feature2transformer(config)
        if ds_train is None and ds_eval is None:
            ds_train = dataFactory.get_train_dataset()
            ds_eval = dataFactory.get_eval_dataset()

        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model factory
        if model is None and modelFunc is not None:
            model = modelFunc(config)
        if model is None:
            raise ValueError("No model found")

        modelFactory = ModelFactoryWithTabular(
            model_args=model_args,
            model=model,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        # compute metrics
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            if labels is None:
                res = self._eval_metric.compute(
                    predictions=logits,
                    references=logits,
                    K=training_args.top_K,
                    candidate_num=training_args.top_K_candidate_num,
                )
                return res
            else:
                res = self._train_metric.compute(
                    prediction_scores=logits, references=labels
                )
                return res

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        trainer = TrainerFactoryWithTabularRecall(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=collator_fn,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(self, model_name_or_path, modelFunc):
        # load data processing
        dataFactory = DataFactoryWithTabular(data_args=None, training_args=None)  # type: ignore
        config = PretrainedConfig.from_pretrained(model_name_or_path)
        dataFactory.load_feature2transformer(config)
        self.feature2trans = dataFactory.FT

        # load model
        self.model = modelFunc(config)
        modelFactory = ModelFactoryWithTabular(model_args=None, model=None)  # type: ignore
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X):
        with torch.no_grad():
            X = self.feature2trans.handle(X)
            for name in X:
                X[name] = torch.tensor([X[name]], dtype=torch.long).to(self.device)  # type: ignore
            return self.model(**X)

    def batch_inference(self, batch_X):
        for _ in range(len(batch_X)):
            batch_X[_] = self.feature2trans.handle(batch_X[_])

        X = {}
        for _ in range(len(batch_X)):
            for name in batch_X[_]:
                if name not in X:
                    X[name] = []
                X[name].append(batch_X[_][name])

        with torch.no_grad():
            for name in X:
                X[name] = torch.tensor(X[name], dtype=torch.long).to(self.device)  # type: ignore
            return self.model(**X)


# Tabular Recall2
class TaskFactoryWithTabularRecall2(TaskFactory):
    def __init__(self, metrics=["metrics/roc_auc", "metrics/topk_recall"]):
        self.TASK = "TaskFactory Tabular recall"
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self._train_metric = evaluate.load(os.path.join(current_dir, metrics[0]))
        self._eval_metric = evaluate.load(os.path.join(current_dir, metrics[1]))

    def create_feature2transformer_and_config(
        self,
        data_args: DataArguments,
        training_args: TrainArguments,
        config: PretrainedConfig,
    ):
        # data factory
        dataFactory = DataFactoryWithTabularRecall2(
            data_args=data_args,
            training_args=training_args,
        )

        dataFactory.create_feature2transformer(config.features)

        obj_dict = {}
        for name, fea in dataFactory.get_feature2meta().items():
            obj_dict[name] = fea.__dict__

        config.features = obj_dict
        config.save_pretrained(os.path.join(data_args.dataset_cache, "meta"))

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        modelFunc: Optional[Type[PreTrainedModel]] = None,
        config: Optional[PretrainedConfig] = None,
        ds_train: Optional[datasets.Dataset] = None,
        ds_eval: Optional[datasets.Dataset] = None,
    ):
        if config is None:
            raise ValueError("No config found")

        # data factory
        dataFactory = DataFactoryWithTabularRecall2(
            data_args=data_args,
            training_args=training_args,
        )
        dataFactory.load_feature2transformer(config)
        if ds_train is None and ds_eval is None:
            ds_train = dataFactory.get_train_dataset()
            ds_eval = dataFactory.get_eval_dataset()

        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model factory
        if model is None and modelFunc is not None:
            model = modelFunc(config)
        if model is None:
            raise ValueError("No model found")

        modelFactory = ModelFactoryWithTabular(
            model_args=model_args,
            model=model,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        # compute metrics
        def compute_metrics_fn(eval_pred):
            if eval_pred.inputs is not None:
                logits, labels, _ = eval_pred
                res = self._eval_metric.compute(
                    predictions=labels,
                    references=logits,
                    K=training_args.top_K,
                )
                return res
            else:
                logits, labels = eval_pred
                res = self._train_metric.compute(
                    prediction_scores=logits, references=labels
                )
                return res

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train

        trainer = TrainerFactoryWithTabularRecall2(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=collator_fn,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(self, model_name_or_path, modelFunc):
        # load data processing
        dataFactory = DataFactoryWithTabular(data_args=None, training_args=None)  # type: ignore
        config = PretrainedConfig.from_pretrained(model_name_or_path)
        dataFactory.load_feature2transformer(config)
        self.feature2trans = dataFactory.FT

        # load model
        self.model = modelFunc(config)
        modelFactory = ModelFactoryWithTabular(model_args=None, model=None)  # type: ignore
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X):
        with torch.no_grad():
            X = self.feature2trans.handle(X)
            for name in X:
                X[name] = torch.tensor([X[name]], dtype=torch.long).to(self.device)  # type: ignore
            return self.model(**X)

    def batch_inference(self, batch_X):
        for _ in range(len(batch_X)):
            batch_X[_] = self.feature2trans.handle(batch_X[_])

        X = {}
        for _ in range(len(batch_X)):
            for name in batch_X[_]:
                if name not in X:
                    X[name] = []
                X[name].append(batch_X[_][name])

        with torch.no_grad():
            for name in X:
                X[name] = torch.tensor(X[name], dtype=torch.long).to(self.device)  # type: ignore
            return self.model(**X)


# Tabular Multilabel Classification
class TaskFactoryWithTabularMultiLabelClassification(TaskFactory):
    def __init__(
        self,
        metrics=[
            "metrics/precision",
            "metrics/recall",
            "metrics/f1",
        ],
    ):
        self.TASK = "TaskFactory Tabular MultiLabel Classification"
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # self._metric = evaluate.combine([os.path.join(current_dir, _) for _ in metrics], "multilabel")
        self._precision_metric = evaluate.load(
            os.path.join(current_dir, "metrics/precision"), "multilabel"
        )
        self._recall_metric = evaluate.load(
            os.path.join(current_dir, "metrics/recall"), "multilabel"
        )
        self._f1_metric = evaluate.load(
            os.path.join(current_dir, "metrics/f1"), "multilabel"
        )

    def create_feature2transformer_and_config(
        self,
        data_args: DataArguments,
        training_args: TrainArguments,
        config: PretrainedConfig,
    ):
        # data factory
        dataFactory = DataFactoryWithTabular(
            data_args=data_args,
            training_args=training_args,
        )

        dataFactory.create_feature2transformer(config.features)

        obj_dict = {}
        for name, fea in dataFactory.get_feature2meta().items():
            obj_dict[name] = fea.__dict__

        config.features = obj_dict
        config.save_pretrained(os.path.join(data_args.dataset_cache, "meta"))

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        modelFunc: Optional[Type[PreTrainedModel]] = None,
        config: Optional[PretrainedConfig] = None,
        ds_train: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
        ds_eval: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
    ):
        if config is None:
            raise ValueError("No config found")

        # data factory
        dataFactory = DataFactoryWithTabular(
            data_args=data_args,
            training_args=training_args,
        )
        dataFactory.load_feature2transformer(config)
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model factory
        if model is None and modelFunc is not None:
            model = modelFunc(config)
        if model is None:
            raise ValueError("No model found")

        modelFactory = ModelFactoryWithTabular(
            model_args=model_args,
            model=model,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        # compute metrics
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            try:
                logits = torch.where(logits >= 0.5, torch.tensor(1), torch.tensor(0))
            except Exception:
                logits = np.where(logits >= 0.5, 1, 0)
            _precision = self._precision_metric.compute(
                predictions=logits, references=labels, average="macro"
            )
            _recall = self._recall_metric.compute(
                predictions=logits, references=labels, average="macro"
            )
            _f1 = self._f1_metric.compute(
                predictions=logits, references=labels, average="macro"
            )
            if _precision is None or _recall is None or _f1 is None:
                return {
                    "precision_macro": 0,
                    "recall_macro": 0,
                    "f1_macro": 0,
                }
            return {
                "precision_macro": _precision["precision"],
                "recall_macro": _recall["recall"],
                "f1_macro": _f1["f1"],
            }

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        trainer = TrainerFactoryWithTabular(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=collator_fn,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(self, model_name_or_path, modelFunc):
        # load data processing
        dataFactory = DataFactoryWithTabular(data_args=None, training_args=None)  # type: ignore
        config = PretrainedConfig.from_pretrained(model_name_or_path)
        dataFactory.load_feature2transformer(config)
        self.feature2trans = dataFactory.FT

        # load model
        self.model = modelFunc(config)
        modelFactory = ModelFactoryWithTabular(model_args=None, model=None)  # type: ignore
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X):
        with torch.no_grad():
            X = self.feature2trans.handle(X)
            for name in X:
                X[name] = torch.tensor([X[name]], dtype=torch.long).to(self.device)  # type: ignore
            return self.model(**X)


# continue pre train factory
class TaskFactoryWithContinuePreTrain(TaskFactory):
    def __init__(self, metrics=["metrics/accuracy"]):
        self.TASK = "TaskFactory continue pre train"

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel],
        config: Optional[PretrainedConfig] = None,
        llm_tokenizer: Optional[PreTrainedTokenizer] = None,
        ds_train: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
        ds_eval: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
    ):
        if model is None:
            raise ValueError("PreTrain need define model")
        # model factory
        modelFactory = ModelFactoryWithContinuePretrain(
            model_args=model_args,
            model=model,
            config=config,
            llm_tokenizer=llm_tokenizer,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # data factory
        dataFactory = DataFactoryWithPretrain(
            data_args=data_args,
            training_args=training_args,
            tokenizer=modelFactory.llm_tokenizer,
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()

        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        #  pretrain no compute metrics

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        trainer = TrainerFactoryWithPretrain(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            data_collator=collator_fn,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )
        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(
        self, tokenizer_name_or_path, model_name_or_path, modelFunc
    ):
        from transformers import AutoTokenizer

        config = PretrainedConfig.from_pretrained(model_name_or_path)
        # load model
        self.model = modelFunc(config)
        llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        modelFactory = ModelFactoryWithPretrain(
            model_args=None,  # type: ignore
            model=self.model,
            config=config,
            llm_tokenizer=llm_tokenizer,  # type: ignore
        )
        self.model = modelFactory._weight_init(self.model, model_name_or_path)
        self.tokenizer = llm_tokenizer
        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X, max_tokens=128):
        with torch.no_grad():
            i = 0
            res = []
            tokens = [-1]
            X = self.tokenizer(X)
            if X.get("input_ids") is None or X.get("attention_mask") is None:
                return "input X is err"

            while i < max_tokens and tokens[0] != self.tokenizer.pad_token_id:
                inputs = {}
                for name in X:
                    if name not in ["input_ids", "attention_mask"]:
                        continue
                    inputs[name] = torch.tensor([X[name]], dtype=torch.long).to(
                        self.device
                    )

                output = self.model(**inputs)
                logits = output.logits
                tokens = torch.argmax(logits, dim=-1)
                tokens = tokens[0].tolist()[:-2:-1]
                X["input_ids"] = X["input_ids"] + tokens  # type: ignore
                X["attention_mask"] = X["attention_mask"] + [1]  # type: ignore
                res += tokens
                i += 1

            answer = self.tokenizer.decode(res)
            return answer


# sft full train factory
class TaskFactoryWithSFTFulltrain(TaskFactory):
    def __init__(self, metrics=["metrics/accuracy"]):
        self.TASK = "TaskFactory sft full train"

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        config: Optional[PretrainedConfig] = None,
        llm_tokenizer: Optional[PreTrainedTokenizer] = None,
        ds_train: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
        ds_eval: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
    ):
        # model factory
        modelFactory = ModelFactoryWithSFTFulltrain(
            model_args=model_args,
            model=model,
            config=config,
            llm_tokenizer=llm_tokenizer,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # data factory
        dataFactory = DataFactoryWithSFT(
            data_args=data_args,
            training_args=training_args,
            tokenizer=modelFactory.llm_tokenizer,
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()

        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        #  pretrain no compute metrics

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        trainer = TrainerFactoryWithPretrain(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            data_collator=collator_fn,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )
        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(
        self, tokenizer_name_or_path, model_name_or_path, modelFunc
    ):
        from transformers import AutoTokenizer

        config = PretrainedConfig.from_pretrained(model_name_or_path)
        # load model
        self.model = modelFunc(config)
        llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        modelFactory = ModelFactoryWithSFTFulltrain(
            model_args=None,  # type: ignore
            model=self.model,
            config=config,
            llm_tokenizer=llm_tokenizer,  # type: ignore
        )
        self.model = modelFactory._weight_init(self.model, model_name_or_path)
        self.tokenizer = llm_tokenizer
        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X, max_tokens=128):
        with torch.no_grad():
            i = 0
            res = []
            tokens = [-1]
            X = self.tokenizer(X)
            if X.get("input_ids") is None or X.get("attention_mask") is None:
                return "input X is err"

            while i < max_tokens and tokens[0] != self.tokenizer.pad_token_id:
                inputs = {}
                for name in X:
                    if name not in ["input_ids", "attention_mask"]:
                        continue
                    inputs[name] = torch.tensor([X[name]], dtype=torch.long).to(
                        self.device
                    )

                output = self.model(**inputs)
                logits = output.logits
                tokens = torch.argmax(logits, dim=-1)
                tokens = tokens[0].tolist()[:-2:-1]
                X["input_ids"] = X["input_ids"] + tokens  # type: ignore
                X["attention_mask"] = X["attention_mask"] + [1]  # type: ignore
                res += tokens
                i += 1

            answer = self.tokenizer.decode(res)
            return answer


# sft lora train factory
class TaskFactoryWithSFTLoratrain(TaskFactory):
    def __init__(self, metrics=["metrics/accuracy"]):
        self.TASK = "TaskFactory sft lora train"

    def train(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainArguments,
        model: Optional[PreTrainedModel] = None,
        config: Optional[PretrainedConfig] = None,
        llm_tokenizer: Optional[PreTrainedTokenizer] = None,
        ds_train: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
        ds_eval: Optional[datasets.Dataset | datasets.DatasetDict | None] = None,
    ):
        # model factory
        modelFactory = ModelFactoryWithSFTLoratrain(
            model_args=model_args,
            model=model,
            config=config,
            llm_tokenizer=llm_tokenizer,
        )
        model = modelFactory.handle()  # type: ignore
        if model is None:
            raise ValueError("No model found")

        # data factory
        dataFactory = DataFactoryWithSFT(
            data_args=data_args,
            training_args=training_args,
            tokenizer=modelFactory.llm_tokenizer,
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()

        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # runtimes
        self._logging_summary(training_args)
        self._load_last_checkpoint(training_args)
        self._seed(training_args)

        #  pretrain no compute metrics

        # data collator
        collator_fn = dataFactory.data_collator_fn
        # train
        training_args.ddp_find_unused_parameters = False  # !!! lora is need for ddp
        trainer = TrainerFactoryWithPretrain(
            model=model,
            args=training_args,
            train_dataset=ds_train,  # type: ignore
            data_collator=collator_fn,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                ),
            ],
        )
        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(ds_train)  # type: ignore
            )
            metrics["train_samples"] = min(max_train_samples, len(ds_train))  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(ds_eval)  # type: ignore
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ds_eval))  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(
        self, tokenizer_name_or_path, model_name_or_path, modelFunc
    ):
        from peft.peft_model import PeftModel
        from transformers import AutoTokenizer

        config = PretrainedConfig.from_pretrained(model_name_or_path)
        # load model
        self.model = modelFunc(config)
        llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        modelFactory = ModelFactoryWithPretrain(
            model_args=None,  # type: ignore
            model=self.model,
            config=config,
            llm_tokenizer=llm_tokenizer,  # type: ignore
        )
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        self.model = PeftModel.from_pretrained(
            model=self.model, model_id=model_name_or_path
        )
        self.tokenizer = llm_tokenizer
        # set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def inference(self, X, max_tokens=128):
        with torch.no_grad():
            i = 0
            res = []
            tokens = [-1]
            X = self.tokenizer(X)
            if X.get("input_ids") is None or X.get("attention_mask") is None:
                return "input X is err"

            while i < max_tokens and tokens[0] != self.tokenizer.pad_token_id:
                inputs = {}
                for name in X:
                    if name not in ["input_ids", "attention_mask"]:
                        continue
                    inputs[name] = torch.tensor([X[name]], dtype=torch.long).to(
                        self.device
                    )

                output = self.model(**inputs)
                logits = output.logits
                tokens = torch.argmax(logits, dim=-1)
                tokens = tokens[0].tolist()[:-2:-1]
                X["input_ids"] = X["input_ids"] + tokens  # type: ignore
                X["attention_mask"] = X["attention_mask"] + [1]  # type: ignore
                res += tokens
                i += 1

            answer = self.tokenizer.decode(res)
            return answer
