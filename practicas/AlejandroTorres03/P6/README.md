# Práctica 6 - Fine-tuning y despliegue de un modelo de clasificación de sentimientos

🔗 **Demo en línea (Hugging Face Space):**  
👉 https://huggingface.co/spaces/Alejandro-03/fine-tuned-sentiment-v1-demo

---

## 📌 Descripción general

Esta práctica consistió en realizar el fine-tuning de un modelo preentrenado (bert-base-uncased) para una tarea de clasificación de sentimientos, utilizando un conjunto de datos de ejemplo. Posteriormente, el modelo ajustado fue subido a Hugging Face Hub para su disponibilidad pública y se integró en una aplicación interactiva desarrollada con Gradio, desplegada en Hugging Face Spaces.

---

## ⚙️ Detalles técnicos de ejecución

- El archivo mi_practica6.ipynb contiene todo el proceso de fine-tuning del modelo, así como el código utilizado para subirlo al repositorio público en Hugging Face, con el identificador:  
  Alejandro-03/fine-tuned-sentiment-v1

- Este modelo entrenado es utilizado directamente en la aplicación (app.py) mediante la función pipeline() de transformers.

- Al ejecutar app.py, la aplicación descarga automáticamente el modelo desde Hugging Face y permite su uso inmediato a través de una interfaz gráfica simple.

- Ahi se genera el link de forma local para la aplicación, sin embargo esta ya se subió a Hugging Face Space, asi que con el link del inicio, se deberia poder ver el modelo en funcionamiento.
---

## 📂 Archivos incluidos

- mi_practica6.ipynb – Notebook con el desarrollo del modelo y su subida a Hugging Face  
- mi_practica6.py – Versión exportada del notebook  
- app.py – Código de la interfaz en Gradio para consumir el modelo  
- requirements.txt – Lista de dependencias necesarias  
- README.md – Este documento  
