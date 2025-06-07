# -*- coding: utf-8 -*-
# Proyecto: Fine-tuning de Transformers con Hugging Face para Análisis de Sentimiento

Esta practica consiste en el entrenamiento y despliegue de un modelo de clasificación de sentimientos usando **Transformers de Hugging Face**. Se entrenó una versión ligera de BERT (`DistilBERT`) con un subconjunto reducido del dataset de reseñas de películas **IMDB**, con el objetivo de realizar *fine-tuning* de forma eficiente.

## 🔗 Enlace al proyecto en Hugging Face Spaces

 [Ver demo en línea](https://huggingface.co/spaces/AldoMZecua/Practica06_NLP)

---

## 📁 Archivos relevantes

- `6_transformers_con_hugginface.ipynb`: Notebook con el código de entrenamiento, evaluación y guardado del modelo.
- `app.py`: Script con una interfaz interactiva creada con **Gradio** para probar el modelo.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

---

## ⚙️ Instalación de dependencias

```bash
pip install -U fsspec
pip install -U datasets
pip install evaluate
pip install -r requirements.txt
