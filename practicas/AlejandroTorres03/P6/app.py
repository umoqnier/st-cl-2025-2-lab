import gradio as gr
from transformers import pipeline

# Carga el modelo fine-tuneado desde tu cuenta
# Reemplaza con TU USUARIO y NOMBRE del modelo en Hugging Face
model_id = "Alejandro-03/fine-tuned-sentiment-v1"

classifier = pipeline("text-classification", model=model_id)

def classify_sentiment(text):
    prediction = classifier(text)[0]
    label = prediction["label"]
    score = round(prediction["score"] * 100, 2)
    return f"{label} ({score}%)"

demo = gr.Interface(
    fn=classify_sentiment,
    inputs="text",
    outputs="text",
    title="Clasificador de Sentimientos",
    description="Escribe una frase en inglés para detectar si el sentimiento es positivo o negativo.",
    theme="default"
)

demo.launch()
