# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Literal, Any
from collections import defaultdict
import torch
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class PairwiseTrainer(Trainer):
    r"""
    Inherits Trainer to compute pairwise loss.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss
        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        Subclass and override to inject custom behavior.

        Note that the first element will be removed from the output tuple.
        See: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py#L3842
        """
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_masks, rejected_masks = torch.split(inputs["attention_mask"], batch_size, dim=0)
        chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)
        chosen_scores = chosen_rewards.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1))
        rejected_scores = rejected_rewards.gather(dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1))
        chosen_scores, rejected_scores = chosen_scores.squeeze(), rejected_scores.squeeze()

        loss = -torch.nn.functional.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()
        if return_outputs:
            return loss, (loss, chosen_scores, rejected_scores)
        else:
            return loss

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")
        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))

            writer.write("\n".join(res))



class LogprobsPairwiseTrainer(Trainer):
    def __init__(
        self, 
        finetuning_args: "FinetuningArguments", 
        processor: Optional["ProcessorMixin"] = None,
        margin: Optional[float] = None, 
        scale_beta: Optional[float] = None,
        auxiliary_coef: Optional[float] = None,
        **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer", None)

        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False
        self.finetuning_args = finetuning_args
        self.can_return_loss = True
        self.margin = margin
        self.scale_beta = scale_beta
        self.auxiliary_coef = auxiliary_coef
        
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))
            
        if finetuning_args.use_badam:
            from types import MethodType
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
            
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()
    
    def _compute_sequence_logprobs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shifted_logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
        shifted_labels = labels[:, 1:]      # [batch, seq_len-1]
        
        loss_mask = (shifted_labels != IGNORE_INDEX).float()
        
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        
        token_log_probs = log_probs.gather(dim=-1, index=shifted_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        
        masked_log_probs = token_log_probs * loss_mask
        
        seq_log_probs = masked_log_probs.sum(dim=-1)
        seq_lengths = loss_mask.sum(dim=-1)
        
        return seq_log_probs, seq_log_probs / (seq_lengths + 1e-8)
    
    def get_batch_logprobs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch["input_ids"].size(0) // 2
        
        outputs = model(**{k: v for k, v in batch.items() if k != "labels"}, return_dict=True)
        logits = outputs.logits
        
        # chosen_input_ids, rejected_input_ids = torch.split(batch["input_ids"], batch_size, dim=0)
        chosen_labels, rejected_labels = torch.split(batch["labels"], batch_size, dim=0)
        chosen_logits, rejected_logits = torch.split(logits, batch_size, dim=0)

        chosen_logprobs, chosen_norm_logprobs = self._compute_sequence_logprobs(
            chosen_logits, chosen_labels
        )
        
        rejected_logprobs, rejected_norm_logprobs = self._compute_sequence_logprobs(
            rejected_logits, rejected_labels
        )
        
        return chosen_logprobs, rejected_logprobs, chosen_norm_logprobs, rejected_norm_logprobs
    
    def get_batch_loss_metrics(
        self, 
        model: "PreTrainedModel", 
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train"
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        
        (
            chosen_logprobs, 
            rejected_logprobs,
            chosen_norm_logprobs, 
            rejected_norm_logprobs,
        ) = self.get_batch_logprobs(model, batch)

        if self.margin is not None:
            logprob_diff = chosen_norm_logprobs - rejected_norm_logprobs - self.margin
        else:
            logprob_diff = chosen_norm_logprobs - rejected_norm_logprobs
        
        if self.scale_beta is not None:
            loss = -torch.nn.functional.logsigmoid(self.scale_beta * logprob_diff).mean()
        else:
            loss = -torch.nn.functional.logsigmoid(logprob_diff).mean()

        sft_loss = -chosen_norm_logprobs.mean()

        if self.auxiliary_coef is not None:
            loss += self.auxiliary_coef * sft_loss
        
        accuracy = (logprob_diff > 0).float().mean()
        
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}logprobs/chosen"] = chosen_norm_logprobs.mean().item()
        metrics[f"{prefix}logprobs/rejected"] = rejected_norm_logprobs.mean().item()
        metrics[f"{prefix}logprobs/diff"] = logprob_diff.mean().item()
        metrics[f"{prefix}logprobs/accuracy"] = accuracy.item()
        metrics[f"{prefix}logprobs/chosen_total"] = chosen_logprobs.mean().item()
        metrics[f"{prefix}logprobs/rejected_total"] = rejected_logprobs.mean().item()
        metrics[f"{prefix}sft_loss"] = sft_loss.item()
        
        return loss, metrics
    
    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", Dict[str, torch.Tensor]]]:
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        
        for key, value in metrics.items():
            self._stored_metrics["train"][key].append(value)
            
        if return_outputs:
            return loss, metrics
        return loss
    
    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())
            
        del self._stored_metrics[train_eval]
        
        if len(metric_list) < 10:
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        
        for key, metric in zip(key_list, metric_list):
            if not key.startswith("dummy_"):
                logs[key] = metric
                
        return super().log(logs, *args, **kwargs)
        
    @override
    def prediction_step(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional["torch.Tensor"], Optional[Tuple["torch.Tensor", "torch.Tensor"]], Optional["torch.Tensor"]]:
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
        
        for key, value in metrics.items():
            self._stored_metrics["eval"][key].append(value)
        
        chosen_logprobs, rejected_logprobs, chosen_norm_logprobs, rejected_norm_logprobs = self.get_batch_logprobs(model, inputs)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, (chosen_norm_logprobs, rejected_norm_logprobs), None)