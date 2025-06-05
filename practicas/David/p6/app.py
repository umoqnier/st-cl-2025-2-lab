import gradio as gr
import json
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
import torch

# Cargar configuración de géneros
with open("labels.json", "r") as f:
    genre2index = json.load(f)
genre_mapping = {v: k for k, v in genre2index.items()}

# Cargar modelo y tokenizador con caché
@gr.cache()
def load_model():
    model = DebertaV2ForSequenceClassification.from_pretrained("davidpmijan/deberta-movies-genres-model")
    tokenizer = DebertaV2Tokenizer.from_pretrained("davidpmijan/deberta-movies-genres-tokenizer")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Función de predicción optimizada
def predict_genres(text, threshold=0.3):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().squeeze()
        
        # Crear diccionario con resultados
        results = {
            "géneros": [],
            "probabilidades": {genre_mapping[i]: float(prob) for i, prob in enumerate(probs)}
        }
        
        # Filtrar géneros sobre el umbral
        results["géneros"] = [genre for genre, prob in results["probabilidades"].items() if prob > threshold]
        
        return results
    except Exception as e:
        return {"error": str(e)}

# Interfaz de Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="Clasificador de Géneros") as demo:
    gr.Markdown("## 🎬 Clasificador de Géneros Cinematográficos")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Descripción de la película", 
                                   placeholder="Ej: Un hacker descubre una conspiración...",
                                   lines=5)
            threshold = gr.Slider(0.0, 1.0, value=0.3, label="Umbral de confianza")
            submit_btn = gr.Button("Predecir géneros")
        
        with gr.Column():
            genres_output = gr.Label(label="Géneros predichos")
            probs_chart = gr.BarPlot(x="género", y="probabilidad", 
                                    title="Probabilidades por género",
                                    width=400, height=300)
    
    # Ejemplos predefinidos
    gr.Examples(
        examples=[
            ["A group of astronauts discover an alien artifact on Mars.", 0.3],
            ["A comedy about two friends starting a business in New York.", 0.2]
        ],
        inputs=[text_input, threshold]
    )
    
    # Evento al hacer clic
    def update_output(text, threshold):
        result = predict_genres(text, threshold)
        if "error" in result:
            raise gr.Error(result["error"])
        
        # Preparar datos para la gráfica
        chart_data = {
            "género": list(result["probabilidades"].keys()),
            "probabilidad": list(result["probabilidades"].values())
        }
        
        return {
            genres_output: ", ".join(result["géneros"]) if result["géneros"] else "No se identificaron géneros",
            probs_chart: chart_data
        }
    
    submit_btn.click(
        fn=update_output,
        inputs=[text_input, threshold],
        outputs=[genres_output, probs_chart]
    )

# Configuración del lanzamiento
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,  # Cambiar a True para enlace público temporal
    favicon_path="🎬"
)