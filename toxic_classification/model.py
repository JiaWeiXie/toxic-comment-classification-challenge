from typing import Any, Callable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class ToxicCommentClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        label_classes: List[str],
        steps_per_epoch: Optional[int] = None,
        n_epochs: int = 3,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.id2label = {idx: label for idx, label in enumerate(label_classes)}
        self.label2id = {label: idx for idx, label in enumerate(label_classes)}
        self.num_labels = len(label_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="multi_label_classification",
        )
        self.f1 = torchmetrics.F1Score(task="multilabel", num_labels=self.num_labels)
        self.t_acc = torchmetrics.Accuracy(
            task="multilabel", num_labels=self.num_labels
        )
        self.t_f1 = torchmetrics.F1Score(task="multilabel", num_labels=self.num_labels)
        self.v_acc = torchmetrics.Accuracy(
            task="multilabel", num_labels=self.num_labels
        )
        self.v_f1 = torchmetrics.F1Score(task="multilabel", num_labels=self.num_labels)

    def forward(self, **inputs: Any) -> Any:
        return self.model(**inputs)

    def training_step(
        self, batch: Tuple[List[int], List[int]], batch_idx: int
    ) -> float:
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.sigmoid(outputs.logits)
        labels = batch["labels"]
        self.log("train_loss", loss)
        self.f1(preds, labels)
        self.log("train_f1", self.f1, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[List[int], List[int]], batch_idx: int) -> float:
        outputs = self(**batch)
        loss, logits = outputs[:2]
        preds = torch.sigmoid(logits)
        labels = batch["labels"]
        self.log("test_loss", loss)
        self.t_acc.update(preds, labels)
        self.t_f1.update(preds, labels)
        return {"loss": loss, "preds": preds, "labels": labels}

    def test_epoch_end(self, outputs):
        self.log(
            "test_acc",
            self.t_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_f1", self.t_f1.compute(), on_step=False, on_epoch=True, prog_bar=True
        )
        self.t_acc.reset()
        self.t_f1.reset()

    def validation_step(
        self, batch: Tuple[List[int], List[int]], batch_idx: int
    ) -> float:
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.sigmoid(logits)
        labels = batch["labels"]
        self.log("val_loss", val_loss)
        self.v_acc.update(preds, labels)
        self.v_f1.update(preds, labels)
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        self.log(
            "valid_acc",
            self.v_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "valid_f1",
            self.v_f1.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.v_acc.reset()
        self.v_f1.reset()

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        return [optimizer], [scheduler]
