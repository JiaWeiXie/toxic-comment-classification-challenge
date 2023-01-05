from typing import List, Dict, Union, Tuple, Any
from difflib import Differ

import pandas as pd
import gradio as gr

from gradio.components import Component

from .service import MLService
from .settings import DATASET_DIR



class MainInterface:
    def __init__(
        self,
        service: MLService,
    ) -> None:
        self.service = service

    def make_inputs(self) -> List[Component]:
        input_text = gr.Textbox(lines=5, max_lines=25, label="Comment text")
        top_k = gr.Slider(minimum=1, maximum=6, step=1, label="Top K")
        threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Threshold")
        over_threshold = gr.Checkbox(label="Filter threshold")
        target_text = gr.Textbox(label="Target labels", visible=False)
        target_count = gr.Number(label="Target count", visible=False)
        input_components = [
            input_text, top_k, threshold,
            over_threshold, target_text, target_count,
        ]
        return input_components

    def make_outputs(self) -> List[Component]:
        result_labels = gr.Label(label="Classification")
        target_labels = gr.HighlightedText(
            label="Diff",
            combine_adjacent=True,
        ).style(color_map={"+": "red", "-": "green"})
        output_components = [result_labels, target_labels]
        return output_components

    def predict(
        self,
        comment_text: str,
        top_k: int,
        threshold: float,
        over_threshold: bool,
        target_text: str,
        target_count: int,
    ) -> Tuple[Dict[str, Union[str, float]], str]:
        data = self.service.predict(
                    comment_text,
                    top_k=top_k,
                    threshold=threshold,
                    over_threshold=over_threshold,
                )
        result_labels = sorted([
            i["label"]
            for i in data
            if i["score"] > threshold
        ])
        result_text = ", ".join(result_labels)
        result_text = f"[ {result_text} ]"
        diff_text = self.diff_texts(target_text, result_text)
        return {i["label"]: i["score"] for i in data}, diff_text

    def diff_texts(self, text1: str, text2: str) -> List[Tuple[str, str]]:
        d = Differ()
        return [
            (token[2:], token[0] if token[0] != " " else None)
            for token in d.compare(text1, text2)
        ]

    def _random_examples(self, df: pd.DataFrame) -> List[List[Any]]:
        examples = []
        for i in range(7):
            sample = df[df["label_count"] == i].sample(n=1, random_state=1).iloc[0]
            labels = self.service.id_to_label(sample["labels"])
            labels_text = ", ".join(sorted(labels))
            label_count = sample["label_count"]
            comment_text = sample["comment_text"]
            examples.append(
                [
                    comment_text,
                    6,
                    0.5,
                    False,
                    f"[ {labels_text} ]",
                    label_count,
                ],
            )
        return examples

    def render(self) -> None:
        gr.close_all()
        df = pd.read_parquet(DATASET_DIR / "train.parquet")
        gr.Interface(
            fn=self.predict,
            inputs=self.make_inputs(),
            outputs=self.make_outputs(),
            title="Toxic Comment Classification",
            examples=self._random_examples(df),
        ).launch(
            debug=True,
            show_error=True,
            server_name="0.0.0.0",
            server_port=8088,
        )
