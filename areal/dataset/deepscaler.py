"""DeepScaleR dataset helpers."""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset

from areal.utils import logging

logger = logging.getLogger(__name__)


def _load_deepscaler_split(path: str, split: Optional[str]):
    target_split = split or "train"
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


def _build_prompt(problem: str) -> str:
    statement = problem.strip()
    if not statement:
        statement = "Solve the problem and provide your final answer."
    return (
        "Solve the following mathematics problem. Show any necessary reasoning and "
        "place the final answer inside \\boxed{}.\n\n"
        f"{statement}"
    )


def get_deepscaler_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = _load_deepscaler_split(path=path, split=split)

    def process(sample, idx):
        prompt = _build_prompt(sample.get("problem", ""))
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


def get_deepscaler_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = _load_deepscaler_split(path=path, split=split)

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

