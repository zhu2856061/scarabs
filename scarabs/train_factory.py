# -*- coding: utf-8 -*-
# @Time   : 2024/08/30 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import, division, print_function

import contextlib
import functools
import os
import sys
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
import torch
from loguru import logger
from prettytable import PrettyTable
from torch.utils.data import RandomSampler
from transformers.trainer import (
    TRAINER_STATE_NAME,
    DataLoader,
    DebugOption,
    DebugUnderflowOverflow,
    DistributedType,
    EvalLoopContainer,
    EvalLoopOutput,
    EvalPrediction,
    ExportableState,
    HPSearchBackend,
    IterableDatasetShard,
    OptimizerNames,
    ParallelMode,
    SaveStrategy,
    Trainer,
    TrainerState,
    TrainOutput,
    accelerate_version,
    deepspeed_init,
    deepspeed_load_checkpoint,
    denumpify_detensorize,
    dist,
    find_batch_size,
    get_model_param_count,
    has_length,
    hp_params,
    is_accelerate_available,
    is_apex_available,
    is_torch_xla_available,
    math,
    nested_concat,
    nested_detach,
    nested_numpify,
    nn,
    shutil,
    skip_first_batches,
    speed_metrics,
    time,
    tpu_spmd_dataloader,
    version,
)
from transformers.trainer_callback import TrainerCallback
from trl import DPOTrainer

if is_apex_available():
    from apex import amp  # type: ignore
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # type: ignore
    import torch_xla.debug.metrics as met  # type: ignore
if is_accelerate_available():
    from accelerate import __version__ as accelerate_version
    from accelerate import skip_first_batches

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]
PREFIX_CHECKPOINT_DIR = "checkpoint"


class PrettyTablePrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just PrettyTable the logs.
    """

    def on_log(self, args, state, control, logs, **kwargs):
        _ = logs.pop("total_flos", None)
        _res = PrettyTable()
        _res.field_names = list(logs.keys())
        _res.add_row(logs.values())
        if state.is_local_process_zero:
            logger.info(f"\n{_res}")


# EvalData Early stop callback function
class EarlyStoppingByEvalDataCallback(TrainerCallback):
    """Determine whether to stop early based on evaluation data
    early_stopping_patience: Judgment frequency，
    early_stopping_threshold: Difference, stop if the condition is not met once within the number of judgments
    """

    def __init__(
        self,
        early_stopping_patience: int = 10,
        early_stopping_threshold: Optional[float] = 1e-7,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        # assert (
        #     args.load_best_model_at_end
        # ), "EarlyStoppingCallback requires load_best_model_at_end = True"
        # assert (
        #     args.metric_for_best_model is not None
        # ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        # assert (
        #     args.evaluation_strategy != IntervalStrategy.NO
        # ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"
        pass

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        # if not metric_to_check.startswith("eval_"):
        #     metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True


# TrainData Early stop callback function
class EarlyStoppingByTrainDataCallback(TrainerCallback):
    """Determine whether to stop early based on train data
    early_stopping_patience: Judgment frequency，
    early_stopping_threshold: Difference, stop if the condition is not met once within the number of judgments
    """

    def __init__(
        self,
        early_stopping_patience: int = 10,
        early_stopping_threshold: Optional[float] = 1e-7,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        # assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        # assert (
        #     args.metric_for_best_model is not None
        # ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        # assert (
        #     args.evaluation_strategy != IntervalStrategy.NO
        # ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"
        pass

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        # if not metric_to_check.startswith("eval_"):
        #     metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True


# class Trainer reconstruction
class TrainerFactory(Trainer):
    """
    transformers==4.47.1 -> opt
    """

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        """
        Cannot be modified to ensure its universality
        """
        if batch_size is None:
            raise ValueError("batch_size is None")
        if args is None:
            raise ValueError("args is None")

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = (
                        self._train_batch_size // max(1, self.args.n_gpu)  # type: ignore
                    )
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size  # type: ignore
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps)
                        * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader) * args.num_train_epochs
                    )
        # Rely on max_steps when dataloader does not have a working size
        elif args.max_steps > 0:
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps)
                    * args.gradient_accumulation_steps
                )
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb  # type: ignore
                for cb in self.callback_handler.callbacks + [self.control]  # type: ignore
                if isinstance(cb, ExportableState)  # type: ignore
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        ################################################################################
        # Train! Settings
        ################################################################################
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer  # type: ignore
        self.callback_handler.lr_scheduler = self.lr_scheduler  # type: ignore
        self.callback_handler.train_dataloader = train_dataloader  # type: ignore

        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)  # type: ignore
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None  # type: ignore
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        ################################################################################
        # Train Start
        ################################################################################

        ####################
        # reconstuct start #
        ####################
        self.train_metrics = None
        ####################
        #  reconstuct end  #
        ####################
        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)  # type: ignore

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )

            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(
                    epoch_dataloader, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            ############################################################################
            # reconstuct start
            ############################################################################
            self.all_preds = None
            self.all_labels = None
            ############################################################################
            # reconstuct end
            ############################################################################

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1

            for _ in range(total_updates):
                update_step += 1
                total_batched_samples += 1
                num_batches = (
                    args.gradient_accumulation_steps
                    if update_step != (total_updates - 1)
                    else remainder
                )
                batch_samples, num_items_in_batch = self.get_batch_samples(
                    epoch_iterator, num_batches
                )
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (
                        step + 1
                    ) % args.gradient_accumulation_steps == 0 or (
                        step + 1
                    ) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(
                            self.model, "main_input_name", "input_ids"
                        )
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(
                                input_tokens, device=self.args.device, dtype=torch.int64
                            )
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()  # type: ignore
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(
                            args, self.state, self.control
                        )
                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type
                        != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    ####################
                    # reconstuct start #
                    with context():
                        out = self.training_step(model, inputs)

                        if isinstance(out, tuple):
                            tr_loss_step, logits, labels = out
                            if logits is not None:
                                logits = self.accelerator.pad_across_processes(
                                    logits, dim=1, pad_index=-100
                                )
                                logits = self.gather_function((logits))
                                self.all_preds = (
                                    logits
                                    if self.all_preds is None
                                    else nested_concat(
                                        self.all_preds, logits, padding_index=-100
                                    )
                                )
                                # For exceeding the length, remove the previous values
                                if (
                                    self.all_preds is not None
                                    and len(self.all_preds) > 100000
                                ):
                                    self.all_preds = self.all_preds[-100000:]

                            if labels is not None:
                                labels = self.accelerator.pad_across_processes(
                                    labels, dim=1, pad_index=-100
                                )
                                labels = self.gather_function((labels))
                                self.all_labels = (
                                    labels
                                    if self.all_labels is None
                                    else nested_concat(
                                        self.all_labels, labels, padding_index=-100
                                    )
                                )
                                # For exceeding the length, remove the previous values
                                if (
                                    self.all_labels is not None
                                    and len(self.all_labels) > 100000
                                ):
                                    self.all_labels = self.all_labels[-100000:]
                        else:
                            tr_loss_step = out
                    #  reconstuct end  #
                    ####################

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (
                            1 + self.state.global_step - self._globalstep_last_logged
                        )
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping
                            if self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type
                                == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()  # type: ignore
                            else:
                                grad_norm = _grad_norm  # type: ignore

                        self.control = self.callback_handler.on_pre_optimizer_step(
                            args, self.state, self.control
                        )

                        self.optimizer.step()  # type: ignore

                        self.control = self.callback_handler.on_optimizer_step(
                            args, self.state, self.control
                        )

                        optimizer_was_run = (
                            not self.accelerator.optimizer_step_was_skipped
                        )
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(
                                self.lr_scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                            ):
                                self.lr_scheduler.step()  # type: ignore

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = (
                            epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        )
                        self.control = self.callback_handler.on_step_end(
                            args, self.state, self.control
                        )
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(
                            args, self.state, self.control
                        )
                    if (
                        self.control.should_epoch_stop
                        or self.control.should_training_stop
                    ):
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(
            self.state.global_step, 0.001
        )  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        ####################
        # reconstuct start #
        ####################
        if self.train_metrics is not None:
            metrics.update(self.train_metrics)
        metrics = {k: round(v, 6) for k, v in metrics.items()}
        ###################
        # reconstuct end  #
        ###################

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
    ):
        """
        Cannot be modified to ensure its universality
        """
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()  # type: ignore

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                6,
            )

            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            ####################
            # reconstuct start #
            if (
                self.compute_metrics is not None
                and self.all_preds is not None
                and self.all_labels is not None
            ):
                tr_metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=self.all_preds,  # type: ignore
                        label_ids=self.all_labels,  # type: ignore
                    )
                )

                preds_mean = self._nested_gather(self.all_preds).float().mean().item()  # type: ignore
                labels_mean = self._nested_gather(self.all_labels).float().mean().item()  # type: ignore

                tr_metrics = denumpify_detensorize(tr_metrics)
                if isinstance(tr_metrics, dict):
                    for key in list(tr_metrics.keys()):
                        tr_metrics[f"train_{key}"] = tr_metrics.pop(key)

                    if preds_mean is not None:
                        tr_metrics.update({"train_preds_mean": preds_mean})
                    if labels_mean is not None:
                        tr_metrics.update({"train_labels_mean": labels_mean})

                    logs.update(tr_metrics)
                    self.train_metrics = tr_metrics

            for k, v in logs.items():
                if k != "learning_rate":
                    logs[k] = round(v, 6)

            is_new_best_metric = self._train_determine_best_metric(
                metrics=logs, trial=trial
            )
            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric
            # reconstuct end  #
            ###################
            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )
            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def _train_determine_best_metric(self, metrics, trial):
        """
        Determine if the model should be saved based on the evaluation metrics.
        If args.metric_for_best_model is not set, the loss is used.

        Returns:
            bool: True if a new best metric was found, else False
        """
        is_new_best_metric = False

        if self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model

            if metric_to_check in metrics:
                metric_value = metrics[metric_to_check]
            else:
                return is_new_best_metric

            operator = np.greater if self.args.greater_is_better else np.less

            if self.state.best_metric is None:
                self.state.best_metric = (
                    float("-inf") if self.args.greater_is_better else float("inf")
                )

            if operator(metric_value, self.state.best_metric):
                run_dir = self._get_output_dir(trial=trial)
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                output_dir = os.path.join(run_dir, checkpoint_folder)

                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                is_new_best_metric = True

        return is_new_best_metric

    # def _determine_best_metric(self, metrics, trial):
    #     """
    #     Determine if the model should be saved based on the evaluation metrics.
    #     If args.metric_for_best_model is not set, the loss is used.

    #     Returns:
    #         bool: True if a new best metric was found, else False
    #     """
    #     is_new_best_metric = False

    #     if self.args.metric_for_best_model is not None:
    #         metric_to_check = self.args.metric_for_best_model

    #         if metric_to_check in metrics:
    #             metric_value = metrics[metric_to_check]
    #         else:
    #             return is_new_best_metric

    #         operator = np.greater if self.args.greater_is_better else np.less

    #         if self.state.best_metric is None:
    #             self.state.best_metric = (
    #                 float("-inf") if self.args.greater_is_better else float("inf")
    #             )

    #         if operator(metric_value, self.state.best_metric):
    #             run_dir = self._get_output_dir(trial=trial)
    #             checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
    #             output_dir = os.path.join(run_dir, checkpoint_folder)

    #             self.state.best_metric = metric_value
    #             self.state.best_model_checkpoint = output_dir

    #             is_new_best_metric = True

    #     return is_new_best_metric

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"  ***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        # if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
        #     self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader  # type: ignore
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_preds = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_labels = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        # all_inputs = EvalLoopContainer(
        #     self.args.eval_do_concat_batches, padding_index=-100
        # )

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=-100
                )
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)  # type: ignore
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            if self.args.batch_eval_metrics:
                if (
                    self.compute_metrics is not None
                    and logits is not None
                    and labels is not None
                ):
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = (
                        losses if "loss" in args.include_for_metrics else None
                    )
                    batch_kwargs["inputs"] = (
                        inputs if "inputs" in args.include_for_metrics else None
                    )
                    metrics = self.compute_metrics(
                        EvalPrediction(
                            predictions=logits,  # type: ignore
                            label_ids=labels,  # type: ignore
                            **batch_kwargs,
                        ),
                        compute_result=is_last_step,  # type: ignore
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)  # type: ignore
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and not self.args.batch_eval_metrics:
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels)  # type: ignore
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = (  # type: ignore
                np.concatenate(all_losses).mean().item()
            )
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()  # type: ignore
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = (  # type: ignore
                self.jit_compilation_time
            )
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = (  # type: ignore
                self.model_preparation_time
            )

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):  # type: ignore
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)  # type: ignore

        ################################################################################
        # reconstuct start
        ################################################################################
        if all_preds is not None:
            metrics[f"{metric_key_prefix}_preds_mean"] = float(np.mean(all_preds))  # type: ignore
        if all_labels is not None:
            metrics[f"{metric_key_prefix}_labels_mean"] = float(np.mean(all_labels))  # type: ignore
        if isinstance(metrics, dict):
            metrics = {k: round(v, 6) for k, v in metrics.items()}
        ################################################################################
        # reconstuct end
        ################################################################################
        return EvalLoopOutput(
            predictions=all_preds,  # type: ignore
            label_ids=all_labels,  # type: ignore
            metrics=metrics,  # type: ignore
            num_samples=num_samples,
        )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ):
        """Module can be modified, and different training modules need to be designed according to different requirements"""
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )

        model.train()
        inputs = self._prepare_inputs(inputs)

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with self.compute_loss_context_manager():
            out = self.compute_loss(
                model,
                inputs,
                return_outputs=self.args.train_return_outputs,  # type: ignore
                num_items_in_batch=num_items_in_batch,
            )
            if isinstance(out, tuple):
                loss, logits = out
            else:
                loss, logits = out, None

        if isinstance(logits, dict):
            logits = tuple(
                v
                for k, v in logits.items()
                if k not in ["loss"]  # type: ignore
            )
        elif isinstance(logits, tuple):
            logits = logits[1:]

        if logits is not None:
            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

        del inputs

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # type: ignore

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps, logits, labels  # type: ignore

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """The module can be modified, and different prediction modules need to be designed according to different requirements"""
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )

                del inputs

                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(
                        v
                        for k, v in outputs.items()
                        if k not in ignore_keys + ["loss"]  # type: ignore
                    )
                else:
                    logits = outputs[1:]

            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys
                    )
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


# LLM Pre training
class TrainerFactoryWithPretrain(TrainerFactory):
    pass


# LLM SFT training
class TrainerFactoryWithSFT(TrainerFactory):
    pass


# LLM DPO
class TrainerFactoryWithDPO(DPOTrainer, TrainerFactory):
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ):
        model.train()

        with self.compute_loss_context_manager():
            loss, metrics = self.compute_loss(
                model,
                inputs,
                return_outputs=True,  # type: ignore
            )
            # self.log(metrics)

        del inputs

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # type: ignore

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return (loss, None, None)

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        # # Build the kwargs for the `map` function
        # with PartialState().local_main_process_first():
        #     dataset = dataset.map(
        #         self.tokenize_row if not self.is_vision_model else self.process_row,
        #         remove_columns=["prompt", "chosen", "rejected"],
        #         fn_kwargs={
        #             "processing_class": processing_class,
        #             "max_prompt_length": args.max_prompt_length,
        #             "max_completion_length": args.max_completion_length,
        #             # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
        #             "add_special_tokens": False,
        #         },
        #         num_proc=args.dataset_num_proc,
        #         desc=f"Tokenizing {dataset_name} dataset",
        #         cache_file_name=f"{args.dataset_cache}/cache/{dataset_name}.tokenize_function",  # type: ignore
        #     )

        return dataset


# Classification
class TrainerFactoryWithLLMClassification(TrainerFactory):
    pass


# Tabular
class TrainerFactoryWithTabular(TrainerFactory):
    pass


# Tabular recall
class TrainerFactoryWithTabularRecall(TrainerFactory):
    pass


# Tabular recall2
class TrainerFactoryWithTabularRecall2(TrainerFactory):
    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False)

        del inputs

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # type: ignore

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps, None, None  # type: ignore

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Cannot be modified to ensure its universality
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"  ***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader  # type: ignore
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_preds = []
        all_labels = []

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # 获得 item embs 矩阵， 并将其喂入 faiss 索引
        with torch.no_grad():
            item_emb_matrix = model.get_item_matrix()
            item_emb_matrix = nested_numpify(item_emb_matrix)
            item_emb_matrix = item_emb_matrix.astype("float32")  # type: ignore
            _, n = item_emb_matrix.shape
            try:
                index = faiss.GpuIndexFlatIP(n)  # 创建 GPU 索引 # type: ignore
                index.add(item_emb_matrix)  # type: ignore
            except Exception:
                index = faiss.IndexFlatIP(n)  # 使用 内积
                index.add(item_emb_matrix)  # type: ignore

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            with torch.no_grad():
                u_emb = model.get_user_embedding(**inputs["uids"])
                u_emb = nested_numpify(u_emb)
                # set to topk=500 here since the retrieval results may contain clicked items
                _, indices = index.search(u_emb, 500)  # type: ignore

                # remove clicked items
                preds = []
                for i in range(len(indices)):
                    new_tmp = []
                    for inx in indices[i]:
                        if inx not in inputs["hist_iids"][i]:
                            new_tmp.append(inx)
                    preds.append(new_tmp)

                all_preds.extend(preds)
                all_labels.extend(inputs["curr_iids"])

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)  # type: ignore
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and not self.args.batch_eval_metrics:
            metrics = self.compute_metrics(
                EvalPrediction(
                    predictions=all_preds,  # type: ignore
                    label_ids=all_labels,  # type: ignore
                    inputs="test",  # type: ignore
                )
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = (  # type: ignore
                self.jit_compilation_time
            )

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):  # type: ignore
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)  # type: ignore

        ################################################################################
        # reconstuct start
        ################################################################################
        if isinstance(metrics, dict):
            metrics = {k: round(v, 6) for k, v in metrics.items()}
        ################################################################################
        # reconstuct end
        ################################################################################
        return EvalLoopOutput(
            predictions=None,  # type: ignore
            label_ids=None,  # type: ignore
            metrics=metrics,  # type: ignore
            num_samples=num_samples,
        )
