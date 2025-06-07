# Práctica 4: Modelos del Lenguaje Neuronales

## Descripción

Esta práctica implementa un modelo del lenguaje neuronal basado en trigramas utilizando PyTorch. El modelo fue entrenado con el corpus `reuters` de NLTK. Posteriormente, se extraen y visualizan los **word embeddings** aprendidos, y se analizan sus propiedades semánticas mediante medidas de similitud coseno y analogías vectoriales.


## Requisitos

- Python 3.10+
- PyTorch
- NLTK
- scikit-learn
- matplotlib
- Google Colab (opcional, recomendado para uso con GPU)
- Montar Google Drive si se desea guardar el modelo



## Cómo correrlo

1. Asegúrate de tener los paquetes instalados:
    ```bash
    pip install torch nltk scikit-learn matplotlib
    ```

2. Descarga recursos necesarios de NLTK:
    ```python
    import nltk
    nltk.download('reuters')
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

3. Ejecuta el script secuencialmente en un entorno como Jupyter o Google Colab.



