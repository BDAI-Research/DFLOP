import torch
from datasets import Dataset
from torch.utils.data import DataLoader


def flush_cache(l2_cache_size: int) -> torch.Tensor:
    cache = torch.empty(l2_cache_size, dtype=torch.int8, device="cuda")
    cache.zero_()
    return cache


def _generate_torchtune_data(num_samples: int, seq_len: int, vocab_size: int) -> Dataset:
    data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (seq_len,)).tolist()
        data.append({"input_ids": input_ids})
    return Dataset.from_list(data)


def _tokenize_torchtune_data(example):
    tokens = example["input_ids"]
    return {"tokens": tokens, "labels": tokens[1:] + [-100]}


def torchtune_loader(num_training_steps: int, batch_size: int, sequence_length: int, vocab_size: int):
    num_samples_for_dummy_data = num_training_steps * batch_size * 2
    dummy_dataset = _generate_torchtune_data(num_samples_for_dummy_data, sequence_length, vocab_size)
    tokenized_dataset = dummy_dataset.map(_tokenize_torchtune_data, batched=False)
    tokenized_dataset.set_format(type="torch", columns=["tokens", "labels"])
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    return iter(dataloader)
