# **Práctica 06** – Fine-tuning en producción  
## **Hecha por:** Diana Laura Salgado Tirado

### 🧠 **Descripción**
En esta práctica se realizó el ajuste fino (*fine-tuning*) de un modelo de lenguaje preentrenado para clasificar oraciones en español como **sexistas** o **no sexistas**. Posteriormente, el modelo fue puesto en producción a través de una interfaz interactiva desarrollada con **Gradio** y desplegada en **Hugging Face Spaces**.

Se utilizó como base el modelo `delarosajav95/HateSpeech-BETO-cased-v2` y el conjunto de datos proveniente de la competencia **EXIST 2024**. 

Además, se usó **CodeCarbon** para medir el impacto ambiental del entrenamiento del modelo.

---

### ⚙️ **Librerías y Dependencias Utilizadas**

Esta práctica hace uso de múltiples librerías para el procesamiento, entrenamiento y despliegue del modelo:

- `pandas`, `numpy`, `os`, `time`, `datetime`
- `transformers` (Hugging Face)
- `datasets`
- `sklearn`
- `torch`
- `evaluate`
- `gradio`
- `huggingface_hub`
- `codecarbon`
- `utils` (archivo auxiliar para el ajuste fino del modelo)

---

### 🌱 **Reporte de Emisiones**
Se utilizó la librería `codecarbon` para estimar las emisiones de CO₂ durante el ajuste fino del modelo, obteniendo un valor aproximado de:

> **0.0041 kg de CO₂**

Este cálculo se realizó en un entorno con GPU (NVIDIA RTX 3080). No fue posible medir las emisiones de la app en producción debido a que Hugging Face Spaces no permite monitorear el uso energético directamente.

---

### 📌 **Notas**
Se emplearon modelos de lenguaje (LLMs) como herramienta de apoyo para la documentación, optimización del código, y corrección ortográfica y sintáctica.

