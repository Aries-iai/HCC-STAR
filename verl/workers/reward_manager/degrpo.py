# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("degrpo")
class DeGRPORewardManager(AbstractRewardManager):
    """Reward manager for DeGRPO, supporting decoupled a_rewards and b_rewards."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", split_char="|||") -> None:
        """
        Initialize the DeGRPORewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
            split_char: The character used to split response into treatment and survival parts for DeGRPO.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.split_char = split_char

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for DeGRPO, returning a_rewards, b_rewards, a_mask, b_mask."""

        # If rm_scores exist, use them directly (fallback for compatibility)
        if "rm_scores" in data.batch.keys() and not return_dict:
            return data.batch["rm_scores"]

        # Initialize outputs
        bs, seq_len = data.batch["responses"].shape
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        a_rewards = torch.zeros(bs, dtype=torch.float32, device=reward_tensor.device)
        b_rewards = torch.zeros(bs, dtype=torch.float32, device=reward_tensor.device)
        a_mask = torch.zeros(bs, seq_len, dtype=torch.float32, device=reward_tensor.device)
        b_mask = torch.ones(bs, seq_len, dtype=torch.float32, device=reward_tensor.device)  # Default: all survival
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # Compute score using custom compute_reward (expects dict with a_rewards, b_rewards, etc.)
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                split_char=self.split_char,
                tokenizer=self.tokenizer
            )

            if isinstance(score, dict):
                # Expect dict with a_rewards, b_rewards, a_mask, b_mask
                a_rewards[i] = score.get("a_rewards", 0.0).item() if isinstance(score.get("a_rewards"), torch.Tensor) else score.get("a_rewards", 0.0)
                b_rewards[i] = score.get("b_rewards", 0.0).item() if isinstance(score.get("b_rewards"), torch.Tensor) else score.get("b_rewards", 0.0)
                if "a_mask" in score and "b_mask" in score:
                    a_mask[i, :valid_response_length] = score["a_mask"][:valid_response_length]
                    b_mask[i, :valid_response_length] = score["b_mask"][:valid_response_length]
                reward = score.get("reward_tensor", a_rewards[i] + b_rewards[i]).item() if isinstance(score.get("reward_tensor"), torch.Tensor) else score.get("reward_tensor", a_rewards[i] + b_rewards[i])
                for key, value in score.get("reward_extra_info", {}).items():
                    reward_extra_info[key].append(value)
            else:
                # Fallback: treat as survival reward
                b_rewards[i] = score
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "a_rewards": a_rewards,
                "b_rewards": b_rewards,
                "a_mask": a_mask,
                "b_mask": b_mask,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
