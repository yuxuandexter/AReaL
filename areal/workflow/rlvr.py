import asyncio
import os
import uuid
from typing import Callable

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger("RLVR workflow")


def default_get_input_ids_fn(data, tokenizer, enable_thinking):
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return input_ids


def default_data_extract_prompt_fn(data):
    return data["messages"]


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        reward_timeout_seconds: float = 1.0,
        get_input_ids_fn: Callable = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        self.async_reward_fn = AsyncRewardWrapper(reward_fn, timeout_seconds=reward_timeout_seconds)
        self.get_input_ids_fn = get_input_ids_fn
        self.data_extract_prompt_fn = data_extract_prompt_fn
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine: InferenceEngine, data):
        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data), self.tokenizer, self.enable_thinking
        )

        # TODO: support llm generation with max_new_token_list
        n_samples = self.gconfig.n_samples
        should_simulate_response = self.gconfig.simulate_response
        requests: list[ModelRequest] = []
        expected_lengths: list[int | None] = []
        if should_simulate_response:
            max_new_token_list = data.get("max_new_token_list")
            if max_new_token_list is None or len(max_new_token_list) == 0:
                raise ValueError(
                    "simulate_response=True requires a non-empty 'max_new_token_list' in the sample data."
                )
            for sample_idx in range(n_samples):
                per_sample_max = int(max_new_token_list[sample_idx % len(max_new_token_list)])
                req = ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(
                        n_samples=1,
                        max_new_tokens=per_sample_max,
                        ignore_eos=True,
                        stop_token_ids=[],
                        stop=[],
                    ),
                    tokenizer=self.tokenizer,
                )
                requests.append(req)
                expected_lengths.append(per_sample_max)
        else:
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            requests.append(req)
            expected_lengths.append(None)
        resps = await asyncio.gather(*[engine.agenerate(req) for req in requests])

        version = engine.get_version()
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []

        results = []
        for resp_idx, resp in enumerate(resps):
            expected_len = (
                expected_lengths[resp_idx]
                if resp_idx < len(expected_lengths)
                else None
            )
            output_len = len(resp.output_tokens)
            # debug for match between expected and actual response length
            # logger.info(
            #     "Simulated response length: expected=%s, actual=%s",
            #     expected_len,
            #     output_len,
            # )
            if expected_len is not None and output_len != expected_len:
                logger.warning(
                    "Simulated response length mismatch: expected %s tokens from max_new_token_list but received %s.",
                    expected_len,
                    output_len,
                )

            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            prompt_strs.append(prompt_str)
            completions_strs.append(completions_str)
            seqlens.append(len(seq))
            reward = await self.async_reward_fn(
                prompt_str,
                completions_str,
                resp.input_tokens,
                resp.output_tokens,
                **data,
            )

            # Log reward.
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

            rewards.append(reward)
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([float(reward)]),
            )
            results.append(res)

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                n_samples = self.gconfig.n_samples
                for i, (p, c, r, sl) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        return concat_padded_tensors(results)