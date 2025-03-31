# -*- coding: utf-8 -*-
# @Time   : 2024/07/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import, division, print_function

import os
import random
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
import transformers
from loguru import logger
from torchinfo import summary
from tqdm import tqdm
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from scarabs.args_factory import (
    DataArguments,
    ModelArguments,
    TaskArguments,
    TrainArguments,
)
from scarabs.data_factory import (
    DataFactoryWithDPO,
    DataFactoryWithLLMClassification,
    DataFactoryWithPretrain,
    DataFactoryWithSFT,
    DataFactoryWithTabular,
)
from scarabs.model_factory import (
    ModelFactoryWithInference,
    ModelFactoryWithLLMClassification,
    ModelFactoryWithPretrain,
    ModelFactoryWithSFTtrain,
    ModelFactoryWithTabular,
)
from scarabs.train_factory import (
    EarlyStoppingByEvalDataCallback,
    EarlyStoppingByTrainDataCallback,
    PrettyTablePrinterCallback,
    TrainerFactoryWithDistill,
    TrainerFactoryWithDPO,
    TrainerFactoryWithGRPO,
    TrainerFactoryWithLLMClassification,
    TrainerFactoryWithPPO,
    TrainerFactoryWithPretrain,
    TrainerFactoryWithSFT,
    TrainerFactoryWithTabular,
)

tqdm.pandas()


# task factory
class TaskFactory:
    TASK = "Abstract TaskFactory Class"

    def __init__(
        self,
        task_args: Optional[TaskArguments] = None,
        data_args: Optional[DataArguments] = None,
        model_args: Optional[ModelArguments] = None,
        training_args: Optional[TrainArguments] = None,
    ):
        self.task_args = task_args
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args

        if self.task_args is not None:
            self.task_name_or_path = self.task_args.task_name_or_path

            # Setup logging
            log_file = os.path.join(self.task_name_or_path, "roll.log")
            logger.add(
                log_file,  # 文件名包含时间戳
                level="INFO",  # 设置最低日志级别
                encoding="utf-8",
                enqueue=True,
            )

    def _logging_summary(self, training_args: TrainArguments):
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

        cpu_count = os.cpu_count()
        if cpu_count is not None:
            torch.set_num_threads(cpu_count // 2)

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

    def inference_with_load_model(
        self,
        tokenizer_name_or_path,
        model_name_or_path,
        modelFunc,
        peft_name_or_path=None,
    ):
        from peft.peft_model import PeftModel
        from transformers import AutoTokenizer

        config = PretrainedConfig.from_pretrained(model_name_or_path)
        # load model
        self.model = modelFunc(config)
        modelFactory = ModelFactoryWithInference(model_args=None, model=self.model)  # type: ignore
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        if peft_name_or_path is not None:
            self.model = PeftModel.from_pretrained(
                model=self.model, model_id=peft_name_or_path
            )

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # set
        def get_best_device():
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 1:  # 在多个GPU的情况下，选择空闲显存最大的GPU
                    max_memory = 0
                    best_device_id = 0
                    for i in range(device_count):
                        memory = torch.cuda.get_device_properties(
                            i
                        ).total_memory - torch.cuda.memory_allocated(i)
                        if memory > max_memory:
                            max_memory = memory
                            best_device_id = i
                    return torch.device(f"cuda:{best_device_id}")
                else:
                    return torch.device("cuda:0")
            else:
                return torch.device("cpu")

        self.device = get_best_device()
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


# Pre train factory
class TaskFactoryWithPreTrain(TaskFactory):
    TASK = "TaskFactory PreTrain"

    def train(
        self, model=None, config=None, tokenizer=None, ds_train=None, ds_eval=None
    ):
        assert self.task_args is not None
        assert self.data_args is not None
        assert self.model_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # model factory
        modelFactory = ModelFactoryWithPretrain(
            model_args=self.model_args,
            model=model,
            config=config,
            llm_tokenizer=tokenizer,
        )
        model = modelFactory.handle()
        assert model is not None

        # data factory
        dataFactory = DataFactoryWithPretrain(
            task_args=self.task_args,
            data_args=self.data_args,
            tokenizer=modelFactory.llm_tokenizer,  # type: ignore
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model show and data show
        if ds_train is not None:
            logger.info("\n Data Check >>>>>>>>>>>>> \n")
            logger.info(ds_train[0])
            logger.info("\n <<<<<<<<<<<<< Data Check \n")
            summary(
                model, depth=20, input_data=dataFactory.data_collator_fn([ds_train[0]])
            )

        # trainer
        trainer = TrainerFactoryWithPretrain(
            model=model,
            args=self.training_args,
            train_dataset=ds_train,  # type: ignore
            data_collator=dataFactory.data_collator_fn,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
        )
        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            metrics["train_samples"] = len(ds_train)  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(ds_eval)  # type: ignore
            metrics["eval_samples"] = len(ds_eval)  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


# sft train factory
class TaskFactoryWithSFTtrain(TaskFactory):
    TASK = "TaskFactory sft full train"

    def train(
        self, model=None, config=None, tokenizer=None, ds_train=None, ds_eval=None
    ):
        assert self.task_args is not None
        assert self.data_args is not None
        assert self.model_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # model factory
        modelFactory = ModelFactoryWithSFTtrain(
            model_args=self.model_args,
            model=model,
            config=config,
            llm_tokenizer=tokenizer,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # data factory
        dataFactory = DataFactoryWithSFT(
            task_args=self.task_args,
            data_args=self.data_args,
            tokenizer=modelFactory.llm_tokenizer,  # type: ignore
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()

        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model show and data show
        if ds_train is not None:
            logger.info("\n Data Check >>>>>>>>>>>>> \n")
            logger.info(ds_train[0])
            logger.info("\n <<<<<<<<<<<<< Data Check \n")
            summary(
                model, depth=20, input_data=dataFactory.data_collator_fn([ds_train[0]])
            )

        # train
        self.training_args.ddp_find_unused_parameters = (
            False  # !!! lora is need for ddp
        )
        trainer = TrainerFactoryWithSFT(
            model=model,
            args=self.training_args,
            train_dataset=ds_train,  # type: ignore
            data_collator=dataFactory.data_collator_fn,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
        )
        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            metrics["train_samples"] = len(ds_train)  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(ds_eval)  # type: ignore
            metrics["eval_samples"] = len(ds_eval)  # type: ignore

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


# dpo train factory
class TaskFactoryWithDPOtrain(TaskFactory):
    TASK = "TaskFactory dpo train"

    def train(
        self,
        model=None,
        ref_model=None,
        config=None,
        tokenizer=None,
        ds_train=None,
        ds_eval=None,
    ):
        assert self.task_args is not None
        assert self.data_args is not None
        assert self.model_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # model factory
        modelFactory = ModelFactoryWithSFTtrain(
            model_args=self.model_args,
            model=model,
            config=config,
            llm_tokenizer=tokenizer,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # data factory
        dataFactory = DataFactoryWithDPO(
            task_args=self.task_args,
            data_args=self.data_args,
            tokenizer=modelFactory.llm_tokenizer,  # type: ignore
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model show and data show
        if ds_train is not None:
            logger.info("\n Data Check >>>>>>>>>>>>> \n")
            logger.info(ds_train[0])
            logger.info("\n <<<<<<<<<<<<< Data Check \n")
            summary(
                model,
                depth=20,
                input_data={
                    "input_ids": dataFactory.data_collator_fn([ds_train[0]])[
                        "prompt_input_ids"
                    ]
                },
            )

        # train
        self.training_args.ddp_find_unused_parameters = (
            False  # !!! lora is need for ddp
        )
        trainer = TrainerFactoryWithDPO(
            model=model,
            ref_model=ref_model,
            args=self.training_args,  # type: ignore
            train_dataset=ds_train,  # type: ignore
            processing_class=modelFactory.llm_tokenizer,
            data_collator=dataFactory.data_collator_fn,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            metrics["train_samples"] = len(ds_train)  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(ds_eval)  # type: ignore
            metrics["eval_samples"] = len(ds_eval)  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


# dpo train factory
class TaskFactoryWithGRPOtrain(TaskFactory):
    TASK = "TaskFactory grpo train"

    def train(
        self,
        model,
        reward_funcs,
        tokenizer,
        ds_train,
        ds_eval,
        peft_config=None,
    ):
        assert self.task_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # train
        self.training_args.ddp_find_unused_parameters = (
            False  # !!! lora is need for ddp  # !!! lora is need for ddp
        )
        trainer = TrainerFactoryWithGRPO(
            model=model,
            reward_funcs=reward_funcs,
            args=self.training_args,  # type: ignore
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            processing_class=tokenizer,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
            peft_config=peft_config,
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            metrics["train_samples"] = len(ds_train)  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(ds_eval)  # type: ignore
            metrics["eval_samples"] = len(ds_eval)  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


# dpo train factory
class TaskFactoryWithPPOtrain(TaskFactory):
    TASK = "TaskFactory ppo train"

    def train(
        self,
        policy,
        ref_policy,
        reward_model,
        value_model,
        tokenizer,
        ds_train,
        ds_eval,
        peft_config=None,
    ):
        assert self.task_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # train
        self.training_args.ddp_find_unused_parameters = (
            False  # !!! lora is need for ddp
        )
        trainer = TrainerFactoryWithPPO(
            args=self.training_args,  # type: ignore
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,
            peft_config=peft_config,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if self.training_args.do_train:
            trainer.train()


# distill train factory
class TaskFactoryWithDistilltrain(TaskFactory):
    TASK = "TaskFactory distill train"

    def train(
        self,
        model=None,
        ref_model=None,
        config=None,
        tokenizer=None,
        ds_train=None,
        ds_eval=None,
    ):
        assert self.task_args is not None
        assert self.data_args is not None
        assert self.model_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # model factory
        modelFactory = ModelFactoryWithSFTtrain(
            model_args=self.model_args,
            model=model,
            config=config,
            llm_tokenizer=tokenizer,
        )
        model = modelFactory.handle()
        assert model is not None

        # data factory
        dataFactory = DataFactoryWithSFT(
            task_args=self.task_args,
            data_args=self.data_args,
            tokenizer=modelFactory.llm_tokenizer,  # type: ignore
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model show and data show
        if ds_train is not None:
            logger.info("\n Data Check >>>>>>>>>>>>> \n")
            logger.info(ds_train[0])
            logger.info("\n <<<<<<<<<<<<< Data Check \n")
            summary(
                model, depth=20, input_data=dataFactory.data_collator_fn([ds_train[0]])
            )

        # train
        self.training_args.ddp_find_unused_parameters = (
            False  # !!! lora is need for ddp
        )
        trainer = TrainerFactoryWithDistill(
            model=model,
            ref_model=ref_model,
            args=self.training_args,  # type: ignore
            train_dataset=ds_train,  # type: ignore
            processing_class=modelFactory.llm_tokenizer,
            data_collator=dataFactory.data_collator_fn,
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByTrainDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            metrics["train_samples"] = len(ds_train)  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(ds_eval)  # type: ignore
            metrics["eval_samples"] = len(ds_eval)  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


# LLM Classification
class TaskFactoryWithLLMClassification(TaskFactory):
    TASK = "TaskFactory LLM text Classification"

    def __init__(
        self,
        task_args: Optional[TaskArguments] = None,
        data_args: Optional[DataArguments] = None,
        model_args: Optional[ModelArguments] = None,
        training_args: Optional[TrainArguments] = None,
        metrics=[
            "metrics/accuracy",
            "metrics/precision",
            "metrics/recall",
            "metrics/f1",
        ],
    ):
        super().__init__(task_args, data_args, model_args, training_args)
        if self.task_args is not None:
            os.makedirs(self.task_args.task_name_or_path, exist_ok=True)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self._metric = evaluate.combine(
                [os.path.join(current_dir, _) for _ in metrics]
            )

    def train(
        self,
        model=None,
        config=None,
        tokenizer=None,
        ds_train=None,
        ds_eval=None,
    ):
        assert self.task_args is not None
        assert self.data_args is not None
        assert self.model_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # model factory
        modelFactory = ModelFactoryWithLLMClassification(
            model_args=self.model_args,
            model=model,
            config=config,
            llm_tokenizer=tokenizer,
        )
        model = modelFactory.handle()
        if model is None:
            raise ValueError("No model found")

        # data factory
        dataFactory = DataFactoryWithLLMClassification(
            task_args=self.task_args,
            data_args=self.data_args,
            tokenizer=modelFactory.llm_tokenizer,  # type: ignore
        )
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # compute metrics
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            logits = logits.argmax(axis=-1)
            res = self._metric.compute(predictions=logits, references=labels)
            return res

        # model show and data show
        if ds_train is not None:
            logger.info("\n Data Check >>>>>>>>>>>>> \n")
            logger.info(ds_train[0])
            logger.info("\n <<<<<<<<<<<<< Data Check \n")
            summary(
                model, depth=20, input_data=dataFactory.data_collator_fn([ds_train[0]])
            )
        # data collator
        # train
        trainer = TrainerFactoryWithLLMClassification(
            model=model,
            args=self.training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=dataFactory.data_collator_fn,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            metrics["train_samples"] = len(ds_train)  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(ds_eval)  # type: ignore
            metrics["eval_samples"] = len(ds_eval)  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference(self, X):
        with torch.no_grad():
            X = self.tokenizer(X)
            for name in X:
                X[name] = torch.tensor([X[name]], dtype=torch.long).to(self.device)
            return self.model(**X)


# Tabular Classification
class TaskFactoryWithTabularClassification(TaskFactory):
    TASK = "TaskFactory Tabular Classification"

    def __init__(
        self,
        task_args: Optional[TaskArguments] = None,
        data_args: Optional[DataArguments] = None,
        model_args: Optional[ModelArguments] = None,
        training_args: Optional[TrainArguments] = None,
        config: Optional[PretrainedConfig] = None,
        metrics=[
            "metrics/accuracy",
            "metrics/precision",
            "metrics/recall",
            "metrics/f1",
        ],
    ):
        super().__init__(task_args, data_args, model_args, training_args)
        self.config = config
        if self.task_args is not None:
            os.makedirs(self.task_args.task_name_or_path, exist_ok=True)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self._metric = evaluate.combine(
                [os.path.join(current_dir, _) for _ in metrics]
            )

    def create_feature2transformer_and_config(self):
        assert self.config is not None
        assert self.task_args is not None
        assert self.data_args is not None

        # data factory
        dataFactory = DataFactoryWithTabular(
            task_args=self.task_args, data_args=self.data_args, config=self.config
        )
        dataFactory.create_feature2transformer()

        obj_dict = {}
        for name, fea in dataFactory.get_feature2meta().items():
            obj_dict[name] = fea.__dict__

        self.config.features = obj_dict
        self.config.save_pretrained(
            os.path.join(
                self.task_args.task_name_or_path, self.data_args.dataset_cache, "meta"
            )
        )

    def train(self, model=None, config=None, ds_train=None, ds_eval=None):
        assert self.config is not None
        assert self.task_args is not None
        assert self.data_args is not None
        assert self.model_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # data factory
        dataFactory = DataFactoryWithTabular(
            task_args=self.task_args, data_args=self.data_args, config=self.config
        )
        dataFactory.load_feature2transformer()
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model factory
        # model factory
        assert model is not None
        modelFactory = ModelFactoryWithTabular(
            model_args=self.model_args,
            model=model,
        )
        model = modelFactory.handle()
        assert model is not None

        # compute metrics
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            logits = logits.argmax(axis=-1)
            res = self._metric.compute(predictions=logits, references=labels)
            return res

        # model show and data show
        if ds_train is not None:
            logger.info("\n Data Check >>>>>>>>>>>>> \n")
            logger.info(ds_train[0])
            logger.info("\n <<<<<<<<<<<<< Data Check \n")
            summary(
                model, depth=20, input_data=dataFactory.data_collator_fn([ds_train[0]])
            )

        # train
        trainer = TrainerFactoryWithTabular(
            model=model,
            args=self.training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=dataFactory.data_collator_fn,
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            metrics["train_samples"] = len(ds_train)  # type: ignore

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(ds_eval)  # type: ignore
            metrics["eval_samples"] = len(ds_eval)  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(self, model_name_or_path, modelFunc):
        # load data processing
        config = PretrainedConfig.from_pretrained(model_name_or_path)
        dataFactory = DataFactoryWithTabular(
            task_args=None, data_args=None, config=config
        )
        dataFactory.load_feature2transformer()
        self.feature2trans = dataFactory.FT

        # load model
        self.model = modelFunc(config)
        modelFactory = ModelFactoryWithTabular(model_args=None, model=None)  # type: ignore
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        # set
        def get_best_device():
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 1:  # 在多个GPU的情况下，选择空闲显存最大的GPU
                    max_memory = 0
                    best_device_id = 0
                    for i in range(device_count):
                        memory = torch.cuda.get_device_properties(
                            i
                        ).total_memory - torch.cuda.memory_allocated(i)
                        if memory > max_memory:
                            max_memory = memory
                            best_device_id = i
                    return torch.device(f"cuda:{best_device_id}")
                else:
                    return torch.device("cuda:0")
            else:
                return torch.device("cpu")

        self.device = get_best_device()
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
    TASK = "TaskFactory Tabular Ctr"

    def __init__(
        self,
        task_args: Optional[TaskArguments] = None,
        data_args: Optional[DataArguments] = None,
        model_args: Optional[ModelArguments] = None,
        training_args: Optional[TrainArguments] = None,
        config: Optional[PretrainedConfig] = None,
        metrics=["metrics/roc_auc"],
    ):
        super().__init__(task_args, data_args, model_args, training_args)

        self.config = config
        if self.task_args is not None:
            os.makedirs(self.task_args.task_name_or_path, exist_ok=True)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self._metric = evaluate.combine(
                [os.path.join(current_dir, _) for _ in metrics]
            )

    def create_feature2transformer_and_config(self):
        assert self.config is not None
        assert self.task_args is not None
        assert self.data_args is not None

        # data factory
        dataFactory = DataFactoryWithTabular(
            task_args=self.task_args, data_args=self.data_args, config=self.config
        )
        dataFactory.create_feature2transformer()

        obj_dict = {}
        for name, fea in dataFactory.get_feature2meta().items():
            obj_dict[name] = fea.__dict__

        self.config.features = obj_dict
        self.config.save_pretrained(
            os.path.join(
                self.task_args.task_name_or_path, self.data_args.dataset_cache, "meta"
            )
        )

    def train(self, model=None, ds_train=None, ds_eval=None):
        assert self.config is not None
        assert self.task_args is not None
        assert self.data_args is not None
        assert self.model_args is not None
        assert self.training_args is not None

        if self.training_args.logging_dir is not None:
            self.training_args.logging_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.logging_dir
            )
        if self.training_args.output_dir is not None:
            self.training_args.output_dir = os.path.join(
                self.task_args.task_name_or_path, self.training_args.output_dir
            )

        # Setup runtimes
        self._logging_summary(self.training_args)
        self._load_last_checkpoint(self.training_args)
        self._seed(self.training_args)

        # data factory
        dataFactory = DataFactoryWithTabular(
            task_args=self.task_args, data_args=self.data_args, config=self.config
        )
        dataFactory.load_feature2transformer()
        if ds_train is None and ds_eval is None:
            ds_train, ds_eval = dataFactory.get_dataset()
        if ds_train is None and ds_eval is None:
            raise ValueError("No dataset found")

        # model factory
        assert model is not None
        modelFactory = ModelFactoryWithTabular(
            model_args=self.model_args,
            model=model,
        )
        model = modelFactory.handle()
        assert model is not None

        # compute metrics
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            res = self._metric.compute(prediction_scores=logits, references=labels)
            return res

        # model show and data show
        if ds_train is not None:
            logger.info("\n Data Check >>>>>>>>>>>>> \n")
            logger.info(ds_train[0])
            logger.info("\n <<<<<<<<<<<<< Data Check \n")
            summary(
                model, depth=20, input_data=dataFactory.data_collator_fn([ds_train[0]])
            )

        # train
        trainer = TrainerFactoryWithTabular(
            model=model,
            args=self.training_args,
            train_dataset=ds_train,  # type: ignore
            eval_dataset=ds_eval,  # type: ignore
            data_collator=dataFactory.data_collator_fn,  # data collator
            compute_metrics=compute_metrics_fn,  # type: ignore
            callbacks=[
                PrettyTablePrinterCallback,  # type: ignore
                EarlyStoppingByEvalDataCallback(
                    self.training_args.early_stopping_patience,
                    self.training_args.early_stopping_threshold,
                ),
            ],
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics
            metrics["train_samples"] = len(ds_train)  # type: ignore
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            assert ds_eval is not None
            metrics = trainer.evaluate(ds_eval)  # type: ignore
            metrics["eval_samples"] = len(ds_eval)  # type: ignore
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    def inference_with_load_model(self, model_name_or_path, modelFunc):
        # load data processing
        config = PretrainedConfig.from_pretrained(model_name_or_path)
        dataFactory = DataFactoryWithTabular(
            task_args=None, data_args=None, config=config
        )
        dataFactory.load_feature2transformer()
        self.feature2trans = dataFactory.FT

        # load model
        self.model = modelFunc(config)
        modelFactory = ModelFactoryWithTabular(model_args=None, model=None)  # type: ignore
        self.model = modelFactory._weight_init(self.model, model_name_or_path)

        # set
        def get_best_device():
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 1:  # 在多个GPU的情况下，选择空闲显存最大的GPU
                    max_memory = 0
                    best_device_id = 0
                    for i in range(device_count):
                        memory = torch.cuda.get_device_properties(
                            i
                        ).total_memory - torch.cuda.memory_allocated(i)
                        if memory > max_memory:
                            max_memory = memory
                            best_device_id = i
                    return torch.device(f"cuda:{best_device_id}")
                else:
                    return torch.device("cuda:0")
            else:
                return torch.device("cpu")

        self.device = get_best_device()
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
