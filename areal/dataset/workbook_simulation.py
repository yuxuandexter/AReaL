"""Lean Workbook simulation dataset helpers."""

from __future__ import annotations
from typing import Optional
from datasets import load_dataset
import json
import random

import torch
from areal.utils import logging

logger = logging.getLogger(__name__)


def _load_workbook_simulation_split(path: str, split: Optional[str]):
    target_split = split or "train"

    if "workbook_simulation" in path:
        path = "Yuxuan13/leanworkbook_with_responses_32k_demo"
    else:
        path = path

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
        statement = "Prove the following theorem in Lean 4."
    return (
        "You are a formal theorem prover. Translate the following problem into "
        "a Lean 4 proof. Show your reasoning and provide the complete proof.\n\n"
        f"{statement}"
    )


def sample_max_new_tokens(n, mean, std, min_val=1, max_val=32768):
    return [max(min(int(random.gauss(mean, std)), max_val), min_val) for _ in range(n)]


def get_workbook_simulation_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = _load_workbook_simulation_split(path=path, split=split)

    def process(sample, idx):
        # Build a rich problem description from available fields
        problem_text = sample.get("problem", "")
        formal_stmt = sample.get("formal_statement", "")
        nl_stmt = sample.get("natural_language_statement", "")

        # Prefer problem, fall back to natural_language_statement
        base_problem = problem_text or nl_stmt or ""

        # Include formal statement as additional context
        if formal_stmt and formal_stmt != base_problem:
            base_problem = f"{base_problem}\n\nFormal statement:\n{formal_stmt}"

        prompt = _build_simulation_prompt(base_problem)
        result = dict(sample)
        result["messages"] = [{"role": "user", "content": prompt}]

        answer = sample.get("answer", "")
        if isinstance(answer, str):
            result["answer"] = answer.strip()

        solution = sample.get("solution", "")
        if isinstance(solution, str):
            result["solution"] = solution.strip()

        query_id = (
            sample.get("id")
            or sample.get("problem_id")
            or sample.get("source_id")
            or sample.get("identifier")
            or f"leanworkbook-{idx}"
        )
        result["query_id"] = query_id

        raw_max_new_token_list = sample.get("max_new_token_list")
        if isinstance(raw_max_new_token_list, str):
            try:
                raw_max_new_token_list = json.loads(raw_max_new_token_list)
            except Exception:
                pass

        if isinstance(raw_max_new_token_list, (list, tuple)) and len(raw_max_new_token_list) > 0:
            result["max_new_token_list"] = [int(v) for v in raw_max_new_token_list]
            logger.info(f"max_new_token_list: {result['max_new_token_list']}")
        else:
            logger.warning(
                "max_new_token_list is not a valid list, using random values"
            )
            n_samples = 10
            mean = 24576
            std = 6144
            max_val = 32768
            result["max_new_token_list"] = sample_max_new_tokens(
                n=n_samples, mean=mean, std=std, max_val=max_val
            )

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


def get_workbook_simulation_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = _load_workbook_simulation_split(path=path, split=split)

    if tokenizer is None:
        raise ValueError("tokenizer is required for SFT dataset preparation.")

    def process(sample):
        problem = sample.get("problem", "").strip()
        # For Lean Workbook, the tactic field contains the proof
        solution = sample.get("tactic") or sample.get("answer") or ""
        solution = solution.strip()

        content = problem
        if solution:
            content = f"{problem}\n\n{solution}"

        seq_tokens = tokenizer.encode(content + tokenizer.eos_token)
        prompt_tokens = tokenizer.encode(problem)
        loss_tokens = [0] * len(prompt_tokens) + [1] * (
            len(seq_tokens) - len(prompt_tokens)
        )

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


def get_simulated_training_batch(
    batch_size: int,
    tokenizer,
    device: torch.device,
    prompt_len: int = 128,
    max_new_token_list: Optional[list[int]] = None,
    pad_token_id: int = 0,
):
    """Generates a simulated training batch using torch.

    The batch contains:
      - input_ids: LongTensor of shape (batch_size, max_seq_len)
      - attention_mask: BoolTensor of shape (batch_size, max_seq_len)
      - loss_mask: BoolTensor of shape (batch_size, max_seq_len)
      - logprobs: FloatTensor of shape (batch_size, max_seq_len)
      - rewards: FloatTensor of shape (batch_size,)
      - values: FloatTensor of shape (batch_size, max_seq_len)

    The sequence length for each sample is prompt_len + new_tokens, where new_tokens
    is sampled from max_new_token_list (or defaults).
    """

    seq_lengths = []

    if max_new_token_list is None:
        n_samples = batch_size
        mean = 24576
        std = 6144
        max_val = 32768
        generated_lengths = sample_max_new_tokens(
            n=n_samples, mean=mean, std=std, max_val=max_val
        )
    else:
        generated_lengths = [random.choice(max_new_token_list) for _ in range(batch_size)]

    for gen_len in generated_lengths:
        total_len = prompt_len + gen_len
        seq_lengths.append(total_len)

    max_seq_len = max(seq_lengths)

    input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    loss_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device=device)
    logprobs = torch.randn((batch_size, max_seq_len), dtype=torch.float, device=device)
    values = torch.randn((batch_size, max_seq_len), dtype=torch.float, device=device)
    rewards = torch.randn((batch_size,), dtype=torch.float, device=device)

    vocab_size = getattr(tokenizer, "vocab_size", 32000)

    for i, length in enumerate(seq_lengths):
        input_ids[i, :length] = torch.randint(0, vocab_size, (length,), device=device)
        attention_mask[i, :length] = 1

        resp_len = length - prompt_len
        if resp_len > 0:
            loss_mask[i, prompt_len:length] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "rewards": rewards,
        "values": values,
    }
