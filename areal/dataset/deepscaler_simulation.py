"""DeepScaleR simulation dataset helpers."""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset

from areal.utils import logging

import random

logger = logging.getLogger(__name__)


def _load_deepscaler_simulation_split(path: str, split: Optional[str]):
    target_split = split or "train"

    # hardcode path to be agentica-org/DeepScaleR-Preview-Dataset for tests
    path = "agentica-org/DeepScaleR-Preview-Dataset"
    try:
        return load_dataset(path=path, split=target_split)
    except ValueError:
        if target_split != "train":
            logger.warning(
                "Split '%s' not found for %s. Falling back to the 'train' split.",
                target_split,
                path,
            )
            return load_dataset(path=path, split="train")
        raise


def _build_simulation_prompt(problem: str) -> str:
    statement = problem.strip()
    if not statement:
        statement = "Solve the problem and provide your final answer."
    return (
        "Solve the following mathematics problem. Show any necessary reasoning and "
        "place the final answer inside \\boxed{}.\n\n"
        f"{statement}"
    )

def sample_max_new_tokens(n, mean, std, min_val=1, max_val=8192):
    return [
        max(min(int(random.gauss(mean, std)), max_val), min_val)
        for _ in range(n)
    ]


def get_deepscaler_simulation_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = _load_deepscaler_simulation_split(path=path, split=split)

    def process(sample, idx):
        prompt = _build_simulation_prompt(sample.get("problem", ""))
        result = dict(sample)
        result["messages"] = [{"role": "user", "content": prompt}]

        answer = sample.get("answer", "")
        if isinstance(answer, str):
            result["answer"] = answer.strip()

        solution = sample.get("solution", "")
        if isinstance(solution, str):
            result["solution"] = solution.strip()

        query_id = (
            sample.get("problem_id")
            or sample.get("id")
            or sample.get("source_id")
            or sample.get("identifier")
            or f"deepscaler-{idx}"
        )
        result["query_id"] = query_id
        # raw_max_new_token_list = sample.get("max_new_token_list")
        # if isinstance(raw_max_new_token_list, (list, tuple)):
        #     result["max_new_token_list"] = [int(v) for v in raw_max_new_token_list]
        # elif raw_max_new_token_list is not None:
        #     result["max_new_token_list"] = [int(raw_max_new_token_list)]
        # else:
        #     # default to a fixed window when metadata is missing
        #     result["max_new_token_list"] = [3072]

        # TODO: replace this with a valid max_new_token_list 
        # hardcode max_new_token_list to be 10 for tests
        # test sharp distirbution:
        # mean: 3072, std: 1024, 512, 256, max_val: 8192
        # test long context distribution:
        # mean: 6144, std: 2048, 512, 256, max_val: 8192
        # test multi-node long context distribution:
        # mean: 12288, std: 2048, max_val: 16384

        n_samples = 10
        mean = 6144
        std = 2048
        max_val = 8192
        result["max_new_token_list"] = sample_max_new_tokens(
            n=n_samples, mean=mean, std=std, max_val=max_val)
        return result

    dataset = dataset.map(process, with_indices=True)

    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when max_length is specified.")

        def filter_length(sample):
            prompt = sample["messages"][0]["content"]
            token_ids = tokenizer.encode(prompt)
            return len(token_ids) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


def get_deepscaler_simulation_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = _load_deepscaler_simulation_split(path=path, split=split)

    if tokenizer is None:
        raise ValueError("tokenizer is required for SFT dataset preparation.")

    def process(sample):
        problem = sample.get("problem", "").strip()
        solution = sample.get("solution") or sample.get("answer") or ""
        solution = solution.strip()

        content = problem
        if solution:
            content = f"{problem}\n\n{solution}"

        seq_tokens = tokenizer.encode(content + tokenizer.eos_token)
        prompt_tokens = tokenizer.encode(problem)
        loss_tokens = [0] * len(prompt_tokens) + [1] * (len(seq_tokens) - len(prompt_tokens))

        result = dict(sample)
        result["input_ids"] = seq_tokens
        result["loss_mask"] = loss_tokens
        return result

    dataset = dataset.map(process)

    if max_length is not None:
        def filter_length(sample):
            return len(sample["input_ids"]) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset

