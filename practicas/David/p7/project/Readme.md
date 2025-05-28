# 🎓 StudyBuddy — Asistente de Estudio basado en RAG

StudyBuddy es un asistente inteligente de estudio diseñado para transformar tus notas académicas en material de aprendizaje efectivo. Utiliza Recuperación Aumentada por Generación (RAG) combinando LangChain, Ollama y Chroma para responder preguntas, generar resúmenes, explicaciones y prácticas de estudio a partir de tus apuntes.

---

## 🚀 Características principales

- ✅ Carga automática de documentos PDF
- ✅ Vectorización e indexación con `ChromaDB` y `OllamaEmbeddings`
- ✅ Chat contextual con soporte para herramientas
- ✅ Interfaz amigable con `Gradio`
- ✅ Soporte para preguntas en lenguaje natural y generación de resúmenes tipo Cornell, mapas mentales y prácticas

---

## 🧱 Estructura del proyecto

studybuddy/
├── app_fixed.py               # App Gradio interactiva
├── StudyBuddy.py              # Clase del asistente inteligente RAG
├── pdfs_notes/                # Directorio para cargar documentos
├── documents_vector_db/       # Base de datos vectorial persistente
├── assets/                    # Avatares (user.png, bot.png)
└── README.md                  # Este archivo

---

## 📦 Requisitos

- Python 3.9+
- Ollama instalado localmente
- Modelos compatibles con tools (`llama3`, `mistral`)
- Dependencias:

pip install -r requirements.txt

## ⚙️ Uso

### 1. Ejecuta Ollama y carga un modelo compatible

ollama run llama3

⚠️ Asegúrate de no usar modelos como `gemma` que no soportan herramientas.

---

### 2. Ejecuta la aplicación

python app_fixed.py

Abre tu navegador en http://localhost:7860.

---

### 3. Usa la app:

- Sube tus notas en PDF
- Haz preguntas como:
  - “Haz un resumen estilo Cornell de este tema”
  - “Genera prácticas de opción múltiple”
  - “Explícame este concepto como si fuera nuevo en el tema”

---

## ❗ Modelos soportados

Solo algunos modelos en Ollama permiten el uso de tools. Usa alguno de los siguientes:

- ✅ llama3
- ✅ mistral
- ❌ gemma (NO compatible)

---

## 🧪 Ejemplos de uso

- “Genera un mapa mental sobre el capítulo 3”
- “Dame 3 preguntas tipo desarrollo con rúbrica de evaluación”
- “Explícame la Ley de Gauss con analogías prácticas”

---

## 📤 Despliegue público (opcional)

Para compartir la app públicamente:

ngrok http 7860

o elimina `server_name="0.0.0.0"` y deja solo `share=True` en `app.launch()`.

---

## 📜 Licencia

Este proyecto se distribuye bajo la licencia MIT.

---

## 🤝 Agradecimientos

Este proyecto utiliza:

- LangChain (https://www.langchain.com/)
- Ollama (https://ollama.com/)
- Gradio (https://gradio.app/)
- Chroma (https://www.trychroma.com/)
