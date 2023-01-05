from typing import Dict, List, Union

import numpy as np
import torch
from transformers import pipeline

from .model import ToxicCommentClassifier
from .settings import BASE_MODEL_NAME, MODEL_PATH


class MLService:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classifier = ToxicCommentClassifier.load_from_checkpoint(MODEL_PATH)
        self.classifier.eval()
        self.pipe = pipeline(
            "text-classification",
            model=self.classifier.model,
            tokenizer=BASE_MODEL_NAME,
            device=self.device,
        )

    def id_to_label(self, labels: List[int]) -> List[str]:
        return [
            self.classifier.id2label[idx]
            for idx, label in enumerate(labels)
            if label == 1
        ]

    def predict(
        self,
        comment_text: str,
        top_k: int = 6,
        threshold: float = 0.5,
        over_threshold: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        outputs = []
        with torch.no_grad():
            outputs = self.pipe(comment_text, top_k=top_k)
            outputs = [
                item
                for item in outputs
                if not over_threshold or item["score"] > threshold
            ]

        return outputs
