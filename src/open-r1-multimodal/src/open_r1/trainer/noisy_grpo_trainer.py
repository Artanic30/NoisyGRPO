# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings
import deepspeed
from peft import PeftModel, LoraConfig
import json
from safetensors import safe_open
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF
import numpy as np

def is_deepspeed_zero2_enabled():
    return deepspeed.runtime.config.zero_optimization_stage == 2

def debug_point():
    import torch.distributed as dist
    if dist.get_rank() == 0:
        import ipdb
        ipdb.set_trace()
    dist.barrier()

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]



class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i: i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count



def load_peft_model(base_model, peft_model_path):
    # 推断 adapter 名称
    def infer_adapter_name(peft_model_path):
        if os.path.exists(os.path.join(peft_model_path, "adapter_config.json")):
            return "default"
        else:
            # 多 adapter 情况
            for subdir in os.listdir(peft_model_path):
                if os.path.exists(os.path.join(peft_model_path, subdir, "adapter_config.json")):
                    return subdir
            raise ValueError("No adapter found in provided path")

    adapter_name = infer_adapter_name(peft_model_path)

    # 加载
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.set_adapter(adapter_name)  # 手动设置 adapter

    return model


def load_peft_config(peft_model_path):
    # Load the adapter config
    adapter_config_path = os.path.join(peft_model_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        peft_config_dict = json.load(f)

    # Convert the dictionary into a LoraConfig object (assuming the JSON contains necessary keys)
    peft_config = LoraConfig(**peft_config_dict
    )

    return peft_config


class VLMNoisyGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        mask_noise_correct: Optional[bool] = False,
        mask_noise_threshold: Optional[float] = 1.0,
        reward_model: str = '',
        inner_gen_bs: Optional[int] = 8,
        cot_reward_var: Optional[float] = 0.11,
        noise_weight:  Optional[float] = 1.0,
        p_value_init: Optional[float] = 1.0,
        r_value: Optional[float] = 0.1,
        p_alpha: Optional[float] = 0.1,
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.beta = args.beta
        self.cot_reward_var = cot_reward_var
        self.inner_gen_bs = inner_gen_bs
        self.noise_weight = noise_weight

        self.x_data = []
        self.y_data = []
        self.iter = 0
        self.p_value = p_value_init
        self.r_value = r_value
        self.p_alpha = p_alpha

        self.vlm_module = vlm_module
        self.mask_noise_correct = mask_noise_correct
        self.mask_noise_threshold = mask_noise_threshold
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype

        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
            # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None



            # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        assert 'noisy' in reward_funcs[-1].__name__, 'noisy reward should be the last reward'

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            pad_token_id=pad_token_id,
        )
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))

        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)

            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[
        str, Union[torch.Tensor, Any]]:

        # prompts = [example["prompt"] for example in inputs]
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        # Handle both pre-loaded images and image paths
        # Only use noisy image when generating the response, not in calculate the reference model loss
        device = self.accelerator.device
        images = []
        prompts = []
        prompts_text = []
        num_noise_list = []
        num_noisy_img = None
        solutions = []
        for x in inputs:
            num_noisy_img = len(x["image"])

            prompts.extend([x["prompt"] for _ in range(num_noisy_img)])
            solutions.extend([x["solution"] for _ in range(num_noisy_img)])
            chat_prompt = self.vlm_module.prepare_prompt(self.processing_class, [x])
            prompts_text.extend([chat_prompt[0] for _ in range(num_noisy_img)])

            for n_img in x["image"]:
                img = n_img["image"]
                # Ensure minimum dimensions of 28 pixels
                w, h = img.size
                if w < 28 or h < 28:
                    # Calculate new dimensions maintaining aspect ratio
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28 / w))
                    else:
                        new_h = 28
                        new_w = int(w * (28 / h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)

                images.append(img)
                num_noise_list.append(n_img['noise_extent'])

        batch_size = int(len(num_noise_list) / num_noisy_img)
        self.num_generations = num_noisy_img

        for i in range(batch_size):
            assert num_noise_list[i * num_noisy_img] == 0.0

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        inner_gen_bs = self.inner_gen_bs
        all_prompt_completion_ids = []
        all_prompt_ids = []
        all_completion_ids = []

        def slice_prompt_inputs(prompt_inputs, start, end):
            sliced = {}
            for k, v in prompt_inputs.items():
                if k == "pixel_values":
                    # 全局的 image_grid_thw 和 patch_counts
                    image_grid_thw = prompt_inputs["image_grid_thw"]  # [B, 3]
                    patch_counts = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]  # [B]

                    # 获取前start个样本的 patch 累积数 -> 起始位置
                    start_idx = patch_counts[:start].sum().item()
                    end_idx = patch_counts[:end].sum().item()

                    sliced[k] = v[start_idx:end_idx]
                else:
                    sliced[k] = v[start:end] if isinstance(v, torch.Tensor) else v
            return sliced

        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            input_len = prompt_ids.size(0)  # total batch size
            for i in range(0, input_len, inner_gen_bs):
                # Get batch slice
                batch_slice = slice(i, min(i + inner_gen_bs, input_len))
                end = min(i + inner_gen_bs, input_len)
                prompt_inputs_b = slice_prompt_inputs(prompt_inputs, i, end)
                prompt_ids_b = prompt_ids[batch_slice]

                generate_returned_result = unwrapped_model.generate(
                    **{k: v for k, v in prompt_inputs_b.items() if k not in self.vlm_module.get_non_generate_params()},
                    generation_config=self.generation_config
                )

                prompt_length = prompt_ids_b.size(1)

                if not self.vlm_module.is_embeds_input():
                    prompt_completion_ids_b = generate_returned_result
                    prompt_ids_b = prompt_completion_ids_b[:, :prompt_length]
                    completion_ids_b = prompt_completion_ids_b[:, prompt_length:]
                else:
                    completion_ids_b = generate_returned_result
                    prompt_completion_ids_b = torch.cat([prompt_ids_b, completion_ids_b], dim=1)

                all_prompt_completion_ids.append(prompt_completion_ids_b)
                all_prompt_ids.append(prompt_ids_b)
                all_completion_ids.append(completion_ids_b)

        def pad_and_cat(tensor_list, pad_token_id):
            # 找到最大长度
            max_len = max(t.size(1) for t in tensor_list)

            padded_tensors = []
            for t in tensor_list:
                pad_len = max_len - t.size(1)
                if pad_len > 0:
                    # 手动 pad 到最大长度
                    pad_tensor = torch.full((t.size(0), pad_len), pad_token_id, dtype=t.dtype, device=t.device)
                    padded = torch.cat([t, pad_tensor], dim=1)
                else:
                    padded = t
                padded_tensors.append(padded)

            return torch.cat(padded_tensors, dim=0)

        pad_token_id = self.generation_config.pad_token_id

        # Concatenate all batches
        prompt_completion_ids = pad_and_cat(all_prompt_completion_ids, pad_token_id)
        prompt_ids = pad_and_cat(all_prompt_ids, pad_token_id)
        completion_ids = pad_and_cat(all_completion_ids, pad_token_id)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        multimodal_inputs = self.vlm_module.convert2clean_image_inputs(multimodal_inputs, num_noisy_img=num_noisy_img)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )

                # with self.accelerator.unwrap_model(self.ref_model).disable_adapters():
            else:
                with self.accelerator.unwrap_model(self.ref_model).disable_adapters():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, **multimodal_inputs)

        rw_per_token_logps = ref_per_token_logps


        logp_attention_mask = attention_mask[:, prompt_length:]


        masked_ref_logps = rw_per_token_logps[:, prompt_length - 1:] * logp_attention_mask

        lengths = logp_attention_mask.sum(dim=1)
        rw_mean_logp = torch.exp(masked_ref_logps.sum(1) / lengths)

        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        trajectory_mask = torch.ones_like(ref_per_token_logps)
        correct_reward = None

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]])
                # assert self.num_generations == 1
                reward_kwargs['noise_extents'] = num_noise_list
                reward_kwargs['solution'] = solutions
                reward_kwargs['noise_reward_logps'] = rw_mean_logp
                reward_kwargs['correct'] = correct_reward
                reward_kwargs['cot_reward_var'] = self.cot_reward_var

                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                if 'correct' in reward_func.__name__:
                    correct_reward = output_reward_func

                if self.mask_noise_correct and 'correct' in reward_func.__name__:
                    clean_reward = [[] for _ in range(batch_size)]
                    for s_idx, (n, r) in enumerate(zip(num_noise_list, output_reward_func)):
                        b_idx = s_idx // num_noisy_img
                        if n == 0:
                            clean_reward[b_idx].append(r)

                    for s_idx, (n, r) in enumerate(zip(num_noise_list, output_reward_func)):
                        b_idx = s_idx // num_noisy_img

                        # if n > 0 and r >= self.mask_noise_threshold:
                        if n > 0 and r > min(clean_reward[b_idx]):
                            trajectory_mask[s_idx] *= 0
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        clean_mask = [1 if _ == 0.0 else 0 for _ in num_noise_list]
        clean_mask_tensor = torch.tensor(clean_mask, dtype=torch.bool)  # 转换为布尔张量
        clean_rewards = rewards_per_func[clean_mask_tensor]  # 只保留 clean_mask 为 1 的行
        clean_rewards = clean_rewards.mean(0)

        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)

        world_size = self.accelerator.num_processes
        # Sum the rewards from all reward functions


        def group_norm(ts):
            ts_reshaped = ts.view(batch_size * world_size, num_noisy_img)

            # Compute grouped-wise rewards
            mean_grouped_ts = ts_reshaped.mean(dim=1)
            std_grouped_ts = ts_reshaped.std(dim=1)
            # Normalize the rewards to compute the advantages
            mean_grouped_ts = mean_grouped_ts.repeat_interleave(num_noisy_img, dim=0)
            std_grouped_ts = std_grouped_ts.repeat_interleave(num_noisy_img, dim=0)
            return (ts - mean_grouped_ts) / (std_grouped_ts + 1e-4), std_grouped_ts, mean_grouped_ts



        with torch.no_grad():
            rw_mean_logp_gathered = self.accelerator.gather(rw_mean_logp)


        rw_mean_logp_gathered, _, _ = group_norm(rw_mean_logp_gathered)

        noise = torch.tensor([1 - _ for _ in num_noise_list]).cuda()
        noise = self.accelerator.gather(noise)
        noise, _, _ = group_norm(noise)
        R = None



        with torch.no_grad():
            reward_semantic = rewards_per_func[:, :-1]
            reward_noise = rewards_per_func[:, -1:]

            reward_semantic_gp, std_semantic_rewards, semantic_mean = group_norm(reward_semantic.sum(dim=1))
            reward_noise_gp, std_noise_rewards, noise_mean = group_norm(reward_noise.sum(dim=1))

            R = self.r_value
            P = self.p_alpha / (std_semantic_rewards ** 2 + self.p_alpha)
            self.p_value = P

            kalman_weight = P / (R + P)

            kalman_weight = torch.clamp(kalman_weight, 0.0, 1.0)
            kalman_weight = kalman_weight.to(reward_semantic_gp.device)
            rewards = reward_semantic_gp + kalman_weight * (reward_noise_gp - reward_semantic_gp)

        advantages, std_grouped_rewards, _ = group_norm(rewards)

        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func)
        reward_per_func = reward_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            if 'correctness' in reward_func_name:
                self._metrics[f"rewards/{reward_func_name}_clean"].append(clean_rewards[i].item())

        self._metrics["mask_noisy_trajectory"].append(1 - (trajectory_mask.sum() / trajectory_mask.numel()).item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        if isinstance(self.p_value, float):
            self._metrics["p_value"].append(self.p_value)

        else:
            self._metrics["p_value"].append(self.accelerator.gather_for_metrics(self.p_value).mean().item())
        self._metrics["kalman_weight"].append(self.accelerator.gather_for_metrics(kalman_weight).mean().item())
        if R is not None:
            if isinstance(R, float):
                self._metrics["R"].append(R)
            else:
                self._metrics["R"].append(self.accelerator.gather_for_metrics(R).mean().item())


        if std_semantic_rewards is not None:
            self._metrics["std_semantic_rewards"].append(
                self.accelerator.gather_for_metrics(std_semantic_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "trajectory_mask": trajectory_mask,
            "per_token_logps": per_token_logps
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        # multimodal_inputs = inputs["multimodal_inputs"]
        trajectory_mask = inputs["trajectory_mask"]
        per_token_logps = inputs["per_token_logps"]

        # Concatenate for full sequence
        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the current policy's log probabilities

        # per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)

        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (
                        ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        per_token_loss = per_token_loss * trajectory_mask
        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
                self.args.per_device_train_batch_size
                * self.accelerator.num_processes
                * self.args.gradient_accumulation_steps
        )

        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
