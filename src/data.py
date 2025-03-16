from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BatchEncoding
from datasets import load_dataset

from .config import BaseConfig


@dataclass
class RedditConfessionsDatasetConfig(BaseConfig):
    batch_size: int = 64
    num_workers: int = 4


class RedditConfessionsDataset(Dataset):
    def __init__(self, tokenizer_path: str = "google-bert/bert-base-uncased", max_length: int = 512):
        self.data = load_dataset("SocialGrep/one-million-reddit-confessions", split="train")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.model_max_length = max_length
        self.pad_token = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.data[idx]["title"], return_tensors="pt", truncation=True)["input_ids"].squeeze(0)

        return tokens[:-1], tokens[1:]

    def collate(self, batch):
        prefix, suffix = zip(*batch)

        return BatchEncoding({
            "prefix": pad_sequence(prefix, batch_first=True, padding_value=self.pad_token),
            "suffix": pad_sequence(suffix, batch_first=True, padding_value=self.pad_token),
            "pad_mask": pad_sequence(
                [torch.zeros(p.shape[0], dtype=torch.bool) for p in prefix],
                batch_first=True,
                padding_value=True
            )
        })


def get_dataloader(
        dataset_config: RedditConfessionsDatasetConfig,
        tokenizer_path: str = "google-bert/bert-base-uncased",
        max_length: int = 512
) -> DataLoader[RedditConfessionsDataset]:
    dataset = RedditConfessionsDataset(
        tokenizer_path, max_length
    )

    return DataLoader(
        dataset,
        batch_size=dataset_config.batch_size,
        num_workers=dataset_config.num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate
    )
