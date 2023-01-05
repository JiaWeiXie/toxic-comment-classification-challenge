import pandas as pd
import plotly.express as px
import streamlit as st

from .service import MLService
from .settings import DATASET_DIR


class MainInterface:
    def __init__(
        self,
        service: MLService,
    ) -> None:
        self.service = service

    def make_form(self) -> None:
        with st.form("input_form"):
            comment_text = st.text_area(
                "Comment text",
                max_chars=512,
                height=120,
            )
            top_k = st.number_input(
                "Top K",
                min_value=1,
                max_value=6,
                value=6,
            )
            over_threshold = st.checkbox("Filter threshold", value=True)
            threshold = st.slider(
                "Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
            submitted = st.form_submit_button("Submit")
            if submitted:
                result_items = []
                with st.spinner("Wait for predict..."):
                    result_items = self.service.predict(
                        comment_text,
                        top_k=top_k,
                        threshold=threshold,
                        over_threshold=over_threshold,
                    )
                    labels_text = ", ".join(
                        [
                            item["label"]
                            for item in result_items
                            if item["score"] > threshold
                        ]
                    )
                    st.markdown(f"Prediction: [ :red[{labels_text}] ]")

                    if result_items:
                        df = pd.DataFrame(result_items)
                        df["text"] = df["score"].apply(lambda x: f"{x * 100:.2f} %")
                        df = df.sort_values(by="score")
                        fig = px.bar(
                            df,
                            x="score",
                            y="label",
                            text="text",
                            orientation="h",
                            width=600,
                        )
                        st.plotly_chart(fig)
                    else:
                        st.info("Not found!")

    def make_examples(self) -> None:
        df = pd.read_parquet(DATASET_DIR / "train.parquet")
        with st.container() as container:
            if st.button("Random examples"):
                self._random_examples(df)
            else:
                self._random_examples(df)

    def _random_examples(self, df: pd.DataFrame) -> None:
        with st.spinner("Wait for random..."):
            for i in range(7):
                sample = df[df["label_count"] == i].sample(n=1, random_state=1).iloc[0]
                labels = self.service.id_to_label(sample["labels"])
                labels_text = ", ".join(labels)
                comment_text = sample["comment_text"]
                st.markdown(f"### {i} lables:\n [ :red[{labels_text}] ]")
                st.markdown(f"{comment_text}")

    def render(self) -> None:
        st.title("Toxic Comment Classification")
        self.make_form()
        self.make_examples()
