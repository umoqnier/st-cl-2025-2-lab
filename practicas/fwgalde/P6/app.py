import gradio as gr
from transformers import pipeline

# Carga del pipeline directamente desde el Hub
classifier = pipeline(
    "text-classification",
    model="fwgalde/dota2-toxic-detector",
    tokenizer="fwgalde/dota2-toxic-detector",
    return_all_scores=True
)

def detect_toxicity(chat: str):
    scores = classifier(chat)[0]
    return {s["label"]: round(s["score"], 3) for s in scores}

iface = gr.Interface(
    fn=detect_toxicity,
    inputs=gr.Textbox(lines=2, placeholder="Escribe un mensaje de chat…"),
    outputs=gr.Label(num_top_classes=3),
    examples=[
        ["gg ez noob team"],
        ["Goog game, well played!"],
        ["Uninstall the game, you're trash"]
    ],
    title="🚨 Detector de Toxicidad para Chats de Videojuegos (Dota 2)",
    description="Clasifica mensajes de chat en **non-toxic**, **mild toxicity** o **toxic**."
)

if __name__ == "__main__":
    iface.launch()
