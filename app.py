"""
app.py — Gradio demo UI for the Sarcasm Detection model.

Launch:
    python app.py --model_dir ./saved_model
    # Opens at http://localhost:7860
"""

import argparse
import gradio as gr
from inference import predict

EXAMPLES = [
    "Scientists Discover Correlation Between Ice Cream Sales and Drowning",
    "Area Man Passionate Defender Of What He Imagines Constitution To Be",
    "New Study Finds That People Who Eat Vegetables Live Longer",
    "Nation's Dogs Vow To Keep Chasing Cars Until They Figure Out What To Do With One",
    "Federal Reserve Raises Interest Rates by 0.25 Percentage Points",
    "God Answers Prayers Of Paralyzed Little Boy; Says No",
]


def classify(headline: str, model_dir: str) -> tuple[str, float, str]:
    if not headline.strip():
        return "—", 0.0, "Please enter a headline."
    result = predict(headline, model_dir)
    label  = f"{'🟠 Sarcastic' if result['label'] == 1 else '🟢 Genuine'}"
    return label, result["confidence"], result["explanation"]


def build_demo(model_dir: str) -> gr.Blocks:
    with gr.Blocks(title="Sarcasm Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🗞️ Sarcasm Detector — News Headlines
            Fine-tuned `bert-base-uncased` · Binary classification · F1 ≥ 0.93
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                headline_box = gr.Textbox(
                    label="News headline",
                    placeholder="Paste a news headline here…",
                    lines=2,
                )
                btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=2):
                label_out      = gr.Textbox(label="Prediction", interactive=False)
                confidence_out = gr.Number(label="Confidence", precision=2)
                explanation_out = gr.Textbox(
                    label="Explanation", interactive=False, lines=3
                )

        gr.Examples(
            examples=EXAMPLES,
            inputs=headline_box,
            label="Try these examples",
        )

        btn.click(
            fn=lambda h: classify(h, model_dir),
            inputs=headline_box,
            outputs=[label_out, confidence_out, explanation_out],
        )
        headline_box.submit(
            fn=lambda h: classify(h, model_dir),
            inputs=headline_box,
            outputs=[label_out, confidence_out, explanation_out],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./saved_model")
    parser.add_argument("--share",     action="store_true")
    parser.add_argument("--port",      type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo(args.model_dir)
    demo.launch(server_port=args.port, share=args.share)
