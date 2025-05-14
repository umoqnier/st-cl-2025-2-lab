import gradio as gr
from transformers import pipeline

# Cargar el modelo desde Hugging Face Hub
model_name = "EARSV/sarcasm-detector"
sarcasm_classifier = pipeline("text-classification", 
                              model=model_name,
                              tokenizer=model_name,
                              return_all_scores=False)

def predict_sarcasm(text):
    # Hacer predicción
    result = sarcasm_classifier(text)[0]
    
    # Formatear salida
    label = "Sarcástico 🔥" if result["label"] == "LABEL_1" else "No sarcástico ✅"
    confidence = f"{result['score']*100:.2f}%"
    
    return label, {  # Devuelve una tupla con dos elementos
        "Confianza": confidence,
        "Etiqueta_original": result["label"],
        "Puntuación": float(result["score"])
    }

# Interfaz Gradio
interface = gr.Interface(
    fn=predict_sarcasm,
    inputs=gr.Textbox(lines=2, placeholder="Escribe un titular aquí...", label="Texto de entrada"),
    outputs=[
        gr.Label(label="Resultado principal"),
        gr.JSON(label="Detalles técnicos")
    ],
    title="Detector de Sarcasmo en Titulares",
    description="Modelo BERT fine-tuned para detectar sarcasmo en titulares de noticias. Entrenado con el dataset Sarcasm_News_Headline",
    examples=[
        ["scientist discovers water is wet, wins nobel prize for groundbreaking revelation"],
        ["new yoga studio opens downtown, offers free classes this weekend"],
        ["government proposes tax on air to fund climate change initiatives"]
    ],
    theme="soft"
)

# Lanzar la aplicación
interface.launch()