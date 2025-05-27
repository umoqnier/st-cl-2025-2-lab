# Práctica 7: Construcción de un *Retrieval-augmented Generation (RAG) especializado* especializado

# StudyBuddy Documentation

## Tutorial: ¡Primeros pasos con StudyBuddy!
1. **Clona el repositorio**:

   ```bash
   git clone https://github.com/fwgalde/st-cl-2025-2-lab.git
   cd practicas/fwgalde/P7
   ```
2. **Instala dependencias**:

   ```bash
   mamba env create --name <entorno> -f requirements.yml
   ```
3. **Descarga modelos** (asegúrate de tener Ollama corriendo):

   ```bash
   ollama pull gemma3:4b
   ```
4. **Inicia la aplicación**:

   ```bash
   streamlit run app.py
   ```
5. **Carga documentos** en la barra lateral (PDF/TXT). Haz clic en **Indexar nuevos docs**. En el caso de prueba utilizamos las notas de clase.
6. **Pregunta** usando la interfaz de chat.

¡Listo! Aprende haciendo tu primera consulta.

---

## How-To: Tareas comunes

### Indexar documentos nuevos

1. Sube archivos desde la barra lateral.
2. Pulsa **Indexar nuevos docs**.
3. Observa el feedback sobre archivos procesados.

### Reiniciar todo el estudio

* Pulsa **Reiniciar estudio** para borrar documentos e índice. La base de conocimiento quedará vacía. Después vas a tener que cerrar la aplicación desde la terminal con `C-c` y ejecutar nuevamente el comando `streamlit run app.py`.

### Ajustar parámetros del modelo

* En la barra lateral, usa el slider **Top k** para cambiar cuántos fragments considera el LLM.
* Modifica variables de entorno `OLLAMA_MODEL` o en `app.py` para cambiar modelo o temperatura.

---

## Explanation: Arquitectura y conceptos

StudyBuddy implementa un sistema **RAG (Retrieval-Augmented Generation)** local:

1. **Ingesta**: carga documentos (TXT/PDF), los fragmenta en trozos de texto.
2. **Embeddings**: convierte fragments en vectores semánticos usando `all-MiniLM-L6-v2`.
3. **Vector Store**: almacena vectores en **ChromaDB** local, persistente en disco.
4. **Recuperación**: ante una pregunta, recupera los *k* fragments más similares.
5. **Generación**: envía contexto + pregunta al LLM (e.g. `gemma3:4b` en Ollama) para obtener la respuesta.


### Componentes principales

* **Streamlit**: interfaz GUI.
* **LangChain**: orquestra loaders, splitter, embeddings, vectorstore y chains.
* **Ollama**: servidor local de LLM.
* **Chroma**: base de datos vectorial ligera.

---

## Reference: Configuración y API

| Parámetro         | Descripción                                  | Valor por defecto     |
| ----------------- | -------------------------------------------- | --------------------- |
| `DOCS_PATH`       | Carpeta para documentos subidos              | `./docs`              |
| `PERSIST_DIR`     | Carpeta de ChromaDB                          | `./db/chroma`         |
| `EMBEDDING_MODEL` | Nombre del modelo de embeddings              | `all-MiniLM-L6-v2`    |
| `LLM_MODEL`       | Identificador del modelo en Ollama           | `gemma3:4b`           |
| `LLM_TEMPERATURE` | Temperatura del muestreo del LLM             | `0.2`                 |
| `CHUNK_SIZE`      | Tamaño de fragmento en caracteres            | `1500`                |
| `CHUNK_OVERLAP`   | Solapamiento entre fragments                 | `300`                 |
| `TOP_K`           | Número de fragments recuperados por consulta | `3` (slider en la UI) |

---

## Reflexión: Limitaciones y consideraciones éticas

Aunque **StudyBuddy** ofrece una potente herramienta de estudio basada en RAG, es importante reconocer sus limitaciones y los posibles riesgos sociales:

1. **Alucinaciones y veracidad**

   * El LLM puede inventar información («hallucinations») al no distinguir claramente entre hechos y sintetizaciones creativas.

2. **Protección de datos y privacidad**

   * Al ingestar documentos personales, existe riesgo de exponer datos sensibles si la app se comparte o se aloja en la nube.

3. **Implicaciones éticas**

   * Dependencia excesiva en IA: los estudiantes podrían sustituir el pensamiento crítico por respuestas automáticas.
   * Desigualdad de acceso: requiere hardware relativamente potente (RAM, CPU) para correr localmente, lo cual puede excluir a usuarios con recursos limitados.

> **Conclusión**: StudyBuddy es una herramienta valiosa para potenciar el aprendizaje, pero debe usarse con **conciencia crítica**, complementada con supervisión humana y prácticas responsables para minimizar riesgos de falta de veracidad, privacidad y exclusión social.

---

## Entorno de desarrollo 🌐
- Fedora Linux 42 (Worstation Edition) 🐧
- Python 3.11.12 🐍
- Jupyter 📓
  - ipywidgets=8.1.7 🎛️
  - jupyter_widget   : 3.0.15
  - jupyter_pygment   : 0.3.0
  - jupyter_server   : 2.27.3 🌐
  - jupyterlab       : 4.4.3 🖥️

## Dependencias 📦
Para el correcto funcionamiento del programa se necesitan las siguientes dependencias:

- chromadb=1.0.9
- langchain=0.3.25
- langchain-chroma=0.2.4
- langchain-community=0.3.24
- langchain-huggingface=0.2.0
- langchain-text-splitters=0.3.8
- loguru=0.7.3
- rich=14.0.0 🎨
- pypdf=5.5.0
- sentence-transformers=4.1.0
- streamlit=1.45.1
- langchain-ollama=0.3.3 ️

Para más información se puede consultar el documento [requirements.yml](requirements.yml) 📄

---

## Integrantes
- [fwgalde](https://github.com/fwgalde)
- [EARSV](https://github.com/EARSV)

## Notas 📝
- Se utilizaron LLM’s 🤖 para la realización de documentación y formato del código. Así como la realización de esta documentación tipo Diátaxis.
- La aplicación para Hugginface fue realizada con el SDK de Gradio y se puede ver el diseño de la "interfaz" en el archivo [app.py](app.py).