from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, BasicTokenizer


class MultiLabelDataset(Dataset):
    def __init__(
        self,
        data: pd.Series,
        labels: List[int],
        tokenizer: BasicTokenizer,
        max_seq_length: int = 512,
    ) -> None:
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.input_ids, self.attention_mask = self._get_features(self.data)

    def _get_features(
        self,
        data: pd.Series,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.tokenizer(
            data.tolist(),
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return features["input_ids"], features["attention_mask"]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }

    def __len__(self) -> int:
        return self.data.count()


class ToxicCommentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        dataset_path: str,
        test_path: str,
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        test_size: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.dataset = pd.read_parquet(
            dataset_path, columns=["comment_text", "labels", "label_w"]
        )
        self.testset = pd.read_parquet(
            test_path, columns=["comment_text", "labels", "label_w"]
        )
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_size = test_size

    def _create_sampler(self, data: pd.Series) -> WeightedRandomSampler:
        data = data.to_numpy()
        return WeightedRandomSampler(
            weights=torch.from_numpy(data).type(torch.double),
            num_samples=len(data),
        )

    def setup(self, stage: str) -> None:
        train_df, val_df = train_test_split(self.dataset, test_size=self.test_size)
        train_labels = train_df["labels"].tolist()
        val_labels = val_df["labels"].tolist()
        test_labels = self.testset["labels"].tolist()
        self.train_sampler = self._create_sampler(train_df["label_w"])
        self.train_dataset = MultiLabelDataset(
            train_df["comment_text"],
            train_labels,
            self.tokenizer,
            self.max_seq_length,
        )
        self.val_dataset = MultiLabelDataset(
            val_df["comment_text"],
            val_labels,
            self.tokenizer,
            self.max_seq_length,
        )
        self.test_dataset = MultiLabelDataset(
            self.testset["comment_text"],
            test_labels,
            self.tokenizer,
            self.max_seq_length,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            sampler=self.train_sampler,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
        )
