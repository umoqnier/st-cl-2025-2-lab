# Práctica 6 de Procesamiento de Lenguaje Natural: Fine Tuning en producción

## Descripción
Esta práctica se centró en utilizar un LLM preentrenado para realizar una tarea de clasificación de reseñas de películas en diferentes géneros. Se utilizó como modelo base DeBerta-v3-small para el fine tuning y el dataset wykonos/movies ambos disponibles en Hugging Face; se utilizó solo 1/3 parte del dataset, se hizo una división entres parte, 75% para entrenamiento, 15% para validación y 10% para test. Se utilizó el optimizador AdamW y se entrenó el modelo durante 8 épocas utilizando early stopping para evitar el sobreajuste. Una vez entrenado el modelo, se subió a Hugging Face y usando Spaces se creó una demo para que los usuarios pudieran probar el modelo. La demo permite a los usuarios ingresar una reseña de película y el modelo devuelve la predicción del género (o géneros) correspondiente.

Adicionalmente, se realizó un seguimiento del consumo de energía durante el entrenamiento del modelo utilizando la biblioteca CodeCarbon. Se registró el consumo de energía y las emisiones de CO2 generadas durante el proceso de entrenamiento. Esto se hizo para evaluar el impacto ambiental del entrenamiento de modelos de lenguaje y para fomentar la conciencia sobre la sostenibilidad en el campo del aprendizaje automático.

## Bibliotecas Utilizadas
- Datasets
- Transformers
- Evaluate
- Pandas
- Torch
- Codecarbon
- Numpy

## Actividades

### 1. **Extracción de Datos**
   - Se utilizó el dataset wykonos/movies de Hugging Face.
   - Se realizó una división del dataset en 75% para entrenamiento, 15% para validación y 10% para test.
   - Se utilizó la función `load_dataset` de la biblioteca Datasets para cargar el dataset y dividirlo.

### 2. **Entrenamiento del Modelo**
   - Se utilizó el modelo DeBerta-v3-small de Hugging Face.
   - Se utilizó el optimizador AdamW y se entrenó el modelo durante 8 épocas.
   - Se implementó early stopping para evitar el sobreajuste.
   - Se utilizó la función `Trainer` de la biblioteca Transformers para facilitar el proceso de entrenamiento

### 3. **Monitoreo del Consumo de Energía**
   - Se utilizó la biblioteca CodeCarbon para registrar el consumo de energía y las emisiones de CO2 generadas durante el entrenamiento del modelo.
   - Se implementó un seguimiento del consumo de energía en tiempo real durante el proceso de entrenamiento.

### 4. **Subida del Modelo a Hugging Face y puesta en producción**
   - Se subió el modelo entrenado a Hugging Face para que estuviera disponible para su uso público.
   - Se creó una demo utilizando Hugging Face Spaces para permitir a los usuarios probar el modelo.
   - La demo permite a los usuarios ingresar una reseña de película y el modelo devuelve la predicción del género (o géneros) correspondiente.


## Requisitos
- Python 3.11
- Las bibliotecas mencionadas anteriormente deben estar instaladas.

## Data
Los datos utilizados para el entrenamiento del modelo de lenguaje provienen de Hugging Face: wykonos/movies
- El dataset contiene reseñas de películas y sus respectivos géneros.

El modelo utilizado es DeBerta-v3-small, que es un modelo de lenguaje preentrenado basado en la arquitectura DeBERTa, disponible en Hugging Face: microsoft/deberta-v3-small.

## Instalación
Para instalar las bibliotecas necesarias, puede utilizar el siguiente comando:
```bash
pip install datasets transformers evaluate pandas torch codecarbon numpy
```

## Ejecución
Para ejecutar los scripts de la práctica, asegúrese de tener todas las bibliotecas instaladas y ejecute los archivos correspondientes desde su entorno de desarrollo.

## Demo
Puede acceder a la demo del modelo en Hugging Face Spaces [aquí](https://huggingface.co/spaces/davidpmijan/movie-genres).
La demo permite a los usuarios ingresar una reseña de película y el modelo devolverá la predicción del género (o géneros) correspondiente.