# Práctica 5 de Procesamiento de Lenguaje Natural: RNN

## Descripción
Esta práctica se centró en la comparación de diferentes enfoques para resolver la tarea de POS Tagging (etiquetado de partes del discurso) desde un método base estadístico hasta un enfoque basado en redes neuronales. Se exploraron diferentes arquitecturas en las redes y técnicas de procesamiento de texto, inciando con una regresión logística y avanzando hacia un modelo de red neuronal feedfoward, después se agregó una capa de embedding y finalmente se implementó un modelo de red neuronal recurrente (RNN).

## Bibliotecas Utilizadas
- Requests
- Rich
- Numpy
- Pandas
- Matplotlib
- Scikit-learn
- Torch
- Gensim
- Scipy

## Actividades

### 1. **Extracción de embeddings de palabras**
- Se seleccionaron 100 palabras al azar del corpus y se obtuvieron sus vectores de representación (embeddings) a partir del modelo RNN trabajado en clase.
- Se utilizaron tanto embeddings estáticos como dinámicos según la arquitectura del modelo.
  
### 2. **Visualización y análisis de agrupamientos**
- Se aplicó un algoritmo de clustering sobre los embeddings de palabras para identificar posibles agrupamientos semánticos o sintácticos.
- Los vectores fueron proyectados a un espacio 2D usando reducción de dimensionalidad (por ejemplo, PCA o t-SNE).
- Se generó un gráfico de dispersión en el que cada punto representa una palabra, coloreado según el cluster al que pertenece.

### 3. **Incorporación de etiquetas POS**
- Al mismo gráfico se agregaron los vectores correspondientes a las etiquetas gramaticales (POS tags), usando marcadores diferenciados para distinguirlos claramente de las palabras.
- Esto permitió observar la relación entre los embeddings de palabras y sus posibles categorías gramaticales.

### 4. **Conclusión**
- Se discutieron los patrones observados en la visualización, evaluando si los clusters obtenidos reflejan relaciones semánticas o gramaticales.
- Se analizó cómo los vectores de etiquetas POS se posicionan respecto a los clusters de palabras.

## Actividad Extra (0.5 pt)

### Implementación de modelo LSTM para POS Tagging
- Se diseñó y entrenó una red basada en *Long Short-Term Memory (LSTM)* para realizar la tarea de etiquetado gramatical (POS).
- Se reportó el desempeño del modelo utilizando la métrica de *accuracy*.
- Se compararon los resultados obtenidos con los del modelo RNN simple utilizado anteriormente.
- Finalmente, se reflexionó sobre cómo la arquitectura LSTM influye en el desempeño del modelo y en su capacidad para capturar dependencias a largo plazo en secuencias de texto.

## Requisitos
- Python 3.10
- Las bibliotecas mencionadas anteriormente deben estar instaladas.

## Data
Los datos utilizados para el entrenamiento del modelo de lenguaje provienen de:
https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/refs/heads/master/es_ancora-ud-train.conllu

## Instalación
Para instalar las bibliotecas necesarias, puede utilizar el siguiente comando:
```bash
pip install nltk numpy pandas matplotlib scikit-learn torch gensim scipy
```

## Ejecución
Para ejecutar los scripts de la práctica, asegúrese de tener todas las bibliotecas instaladas y ejecute los archivos correspondientes desde su entorno de desarrollo.