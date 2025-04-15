# Práctica de Procesamiento de Lenguaje Natural

## Descripción
Esta práctica se centró en entender los modelos neuronales del lenguaje, algunas de sus ventajas y su lugar en el proceso del marco de trabaajo de PLN. Se utilizaron diversas bibliotecas de Python para llevar a cabo las actividades propuestas.

## Bibliotecas Utilizadas
- NLTK
- Numpy = 1.26.4
- Pandas
- Matplotlib
- Scikit-learn
- Torch
- Gensim

## Actividades

1. **Análisis de embeddings dinámicos con clustering jerárquico**  
   - Se extrajeron los *embeddings* dinámicos de 100 palabras del vocabulario utilizado durante el entrenamiento de una RNN para la tarea de *POS tagging*.  
   - Se aplicó un análisis de *clustering jerárquico* para visualizar y explorar la agrupación semántica de las palabras en función de sus representaciones contextuales.  
   - Se graficó el dendrograma resultante para interpretar las relaciones entre las palabras en el espacio de embedding.

2. **Entrenamiento de un modelo LSTM para POS tagging**  
   - Se entrenó un segundo modelo, esta vez utilizando una arquitectura LSTM, para abordar la misma tarea de *POS tagging*.  
   - Se comparó el rendimiento del modelo LSTM con el de la RNN previamente entrenada, evaluando métricas como la precisión y la pérdida.  
   - El objetivo fue observar el impacto de una arquitectura más avanzada sobre la calidad del etiquetado y la representación de palabras.

## Requisitos
- Python 3.10
- Las bibliotecas mencionadas anteriormente deben estar instaladas.

## Instalación
Para instalar las bibliotecas necesarias, puede utilizar el siguiente comando:
```bash
pip install nltk numpy pandas matplotlib scikit-learn torch
```

## Ejecución
Para ejecutar los scripts de la práctica, asegúrese de tener todas las bibliotecas instaladas y ejecute los archivos correspondientes desde su entorno de desarrollo.

## Contribuciones
Las contribuciones a este proyecto son bienvenidas. Por favor, cree un fork del repositorio y envíe un pull request con sus mejoras.