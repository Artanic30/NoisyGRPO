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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMNoisyGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
from open_r1.rewards import reward_funcs_registry
import numpy as np

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, \
    apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from torchvision.transforms import ToTensor, ToPILImage


import spacy
import string
from sentence_transformers import SentenceTransformer, util



def debug_point():
    import torch.distributed as dist
    if dist.get_rank() == 0:
        import ipdb
        ipdb.set_trace()
    dist.barrier()


def random_mask(image, mask_ratio):
    """
    对输入图片进行随机mask，mask掉 n_r% 的像素点，并使用全局平均颜色填充。

    :param image: PIL.Image, 输入RGB图像
    :param mask_ratio: float, 需要mask的比例（0~1之间）
    :return: PIL.Image, 经过mask后的图像
    """
    image = np.array(image)  # 转换为 NumPy 数组
    h, w, c = image.shape
    num_mask = int(h * w * mask_ratio)  # 计算需要mask的像素数量

    # 计算全局平均颜色（对每个通道求平均）
    avg_color = image.reshape(-1, c).mean(axis=0).astype(np.uint8)

    # 生成随机 mask 索引
    mask_indices = np.random.choice(h * w, num_mask, replace=False)
    mask = np.ones((h, w), dtype=bool)
    mask.flat[mask_indices] = False  # 置 0 进行mask

    # 应用 mask（用平均颜色填充）
    masked_image = image.copy()
    masked_image[~mask] = avg_color  # 替换为平均颜色

    return Image.fromarray(masked_image)


def random_mask_patch(image, mask_ratio, num_patches=8):
    """
    对输入图片进行随机 patch 级别的 mask，遮挡 mask_ratio 的 patch，并用全局平均颜色填充。

    :param image: PIL.Image, 输入RGB图像
    :param mask_ratio: float, 需要mask的比例（0~1之间）
    :param num_patches: int, 图片切分成 num_patches x num_patches 个小块
    :return: PIL.Image, 经过mask后的图像
    """
    image = np.array(image)
    h, w, c = image.shape
    patch_h, patch_w = h // num_patches, w // num_patches

    # 计算全局平均颜色（对每个通道求平均）
    avg_color = image.reshape(-1, c).mean(axis=0).astype(np.uint8)

    # 计算总 patch 数量 & 需要 mask 的 patch 数量
    total_patches = num_patches * num_patches
    num_mask = int(total_patches * mask_ratio)

    # 生成随机 mask 索引
    mask_indices = np.random.choice(total_patches, num_mask, replace=False)

    # 遍历所有 patch 进行 mask
    masked_image = image.copy()
    for idx in mask_indices:
        row, col = divmod(idx, num_patches)
        y1, y2 = row * patch_h, (row + 1) * patch_h
        x1, x2 = col * patch_w, (col + 1) * patch_w
        masked_image[y1:y2, x1:x2] = avg_color  # 用平均颜色填充

    return Image.fromarray(masked_image)


def add_diffusion_noise_pil(image_pil, noise_step, num_steps=1000):
    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    # Convert PIL Image to Tensor
    image_tensor = ToTensor()(image_pil).unsqueeze(0)  # Add batch dimension (B, C, H, W)

    noise_delta = int(noise_step)  # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)

    # Convert tensor back to PIL Image
    image_pil_cd = ToPILImage()(image_tensor_cd.squeeze(0))  # Remove batch dimension (C, H, W)

    return image_pil_cd



def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["correctness", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    noise_extents: list[float] = field(
        default_factory=list,
        metadata={"help": "Root directory of the image"},
    )
    noise_type: Optional[str] = field(
        default='gaussian_noise',
        metadata={"help": "Root directory of the image"},
    )
    mask_noise_correct: Optional[bool] = field(
        default=False,
        metadata={"help": "mask the sample that answer correct answers with noisy image"},
    )
    mask_noise_threshold: Optional[float] = field(
        default=1.0,
        metadata={"help": "mask the sample that answer correct answers with noisy image"},
    )
    noise_decaying: Optional[bool] = field(
        default=False,
        metadata={"help": "mask the sample that answer correct answers with noisy image"},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    decay_ratio: Optional[float] = field(
        default=0.5,
        metadata={"help": "mask the sample that answer correct answers with noisy image"},
    )
    inner_gen_bs: Optional[int] = field(
        default=8,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    cot_reward_var: Optional[float] = field(
        default=0.15,
    )
    noise_weight: Optional[float] = field(
        default=1.0,
    )
    r_value: Optional[float] = field(
        default=0.1,
    )
    p_alpha: Optional[float] = field(
        default=0.01,
    )
    question_template: Optional[str] = field(
        default='cot',
        metadata={"help": "use math template"},
    )




@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template
        self.iteration = 0

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def decay_noise_extent(self, noise_extents, iteration):
        """
        Linearly decay the noise_extents to half of their initial values over total_steps iterations.

        Args:
            noise_extents (list): List of initial noise extent values
            iteration (int): Current iteration (0 <= iteration <= total_steps)
            total_steps (int): Total number of steps/iterations for decay

        Returns:
            list: Decayed noise extents for the current iteration
        """
        total_steps = self.__len__() // torch.distributed.get_world_size()
        if total_steps <= 0:
            return noise_extents.copy()

        if iteration > total_steps:
            print(f'total steps and iteration unmatched!!!!')

        # Calculate decay factor (1.0 at start, 0.5 at end)
        decay_factor = 1.0 - 0.5 * min(iteration / total_steps, 1.0)

        # Apply decay to each noise extent
        decayed_extents = [extent * decay_factor for extent in noise_extents]

        return decayed_extents

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        
        QUESTION_TEMPLATE = self.question_template
        def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
        else:
            image = None

        noisy_image_list = []
        noise_extents = self.script_args.noise_extents
        noise_type = self.script_args.noise_type
        if self.script_args.noise_decaying:
            noise_extents = self.decay_noise_extent(noise_extents, self.iteration)

        assert noise_extents[0] == 0.0
        for n_r in noise_extents:
            if noise_type == 'gaussian_noise':
                noisy_image = add_diffusion_noise_pil(image, int(n_r * 1000), 1000)
            elif noise_type == 'random_mask':
                noisy_image = random_mask(image, mask_ratio=n_r)
            elif noise_type == 'random_patch_mask':
                noisy_image = random_mask_patch(image, mask_ratio=n_r)
            else:
                raise NotImplementedError
            noisy_image_list.append({
                'image': noisy_image,
                'noise_extent': n_r
            })

        self.iteration += 1
        assert 'image' in example, "Do not support pure text training."
            
        return {
            'image': noisy_image_list,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # Load the reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type=script_args.question_template))

    print(f'script_args: {script_args}')
    trainer_cls = VLMNoisyGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        mask_noise_correct=script_args.mask_noise_correct,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
        reward_model=script_args.reward_model,
        inner_gen_bs=script_args.inner_gen_bs,
        cot_reward_var=script_args.cot_reward_var,
        noise_weight=script_args.noise_weight,
        p_alpha=script_args.p_alpha,
        r_value=script_args.r_value
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
