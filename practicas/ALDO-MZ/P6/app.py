from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast
import gradio as gr
import torch

# Cargar el modelo y tokenizer guardados
model_path = "./sentiment_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Mover a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Crear pipeline de análisis de sentimiento
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def analyze_sentiment(text):
    # Realizar predicción
    result = classifier(text)[0]
    
    # Formatear resultado
    label = "Positive" if result['label'] == "LABEL_1" else "Negative"
    score = result['score']
    
    return {
        "Sentimiento": label,
        "Confianza": float(score),
        "Positive" if label == "Positive" else "Negative": float(score)
    }

# Interfaz Gradio
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Escribe tu reseña aquí...", label="Texto"),
    outputs=[
        gr.Label(num_top_classes=2, label="Predicción"),
        gr.Textbox(label="Análisis detallado")
    ],
    title="Analizador de Sentimiento",
    description="Clasifica reseñas de películas como positivas o negativas.",
    examples=[
        ["This movie was absolutely fantastic! The acting was brilliant."],
        ["I hated every minute of this terrible film."],
        ["It was okay, not great but not awful either."]
    ],
    theme="soft"
)

# Añadir interpretación
demo.launch(share=True)  # share=True crea un enlace público temporal