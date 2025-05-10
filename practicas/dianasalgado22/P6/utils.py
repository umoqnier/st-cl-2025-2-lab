# Librerias
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
import numpy as np
from evaluate import load
import time
import datetime
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
from os.path import join

# Limpieza

def prepareDatasets(train_rute: str, test_rute: str):
    """
    Combina dos archivos CSV (entrenamiento y prueba) para crear un nuevo conjunto de entrenamiento más grande
    y divide los datos en conjuntos de entrenamiento, validación y prueba.

    Parameters
    ----------
    train_rute : str
        Ruta del archivo CSV de entrenamiento.
    test_rute : str
        Ruta del archivo CSV de prueba.

    Returns
    -------
    tuple
        Tupla que contiene tres objetos Dataset (entrenamiento, validación y prueba).
    
    Notes
    -----
    - Se eliminan las filas con etiqueta neutra (-1).
    - Se combinan los datos de entrenamiento y prueba.
    - Se dividen los datos en 80% entrenamiento, 10% validación, y 10% prueba.
    - Se renombran las columnas para que sean compatibles con el formato de `datasets`.
    """
    # Convertir a dataframes
    df_train = pd.read_csv(train_rute)
    df_test = pd.read_csv(test_rute)

    # Eliminar datos con etiqueta neutra (-1)
    df_train = df_train[df_train['hard_label_t1'] != -1]
    df_test = df_test[df_test['hard_label_t1'] != -1]

    # Concatenar los datos de entrenamiento y validación
    # Para tener una mayor cantidad de datos de entrenamiento

    # Seleccionamos solo las columnas relevantes: id, tweet limpio y la etiqueta 'hard_label_t1
    df_concatenated = pd.concat([df_train[['id_EXIST','clean_tweet','hard_label_t1']],
                                 df_test[['id_EXIST','clean_tweet','hard_label_t1']]])
    
    # Dividir los datos en un gran conjunto de entrenamiento (80%) y el resto (20%) 
    big_train, not_train = train_test_split(df_concatenated, test_size=0.2, random_state=42)

    # Dividir el 20% restante de los datos en validación (10%) y prueba (10%)
    valid, test = train_test_split(not_train, test_size=0.5, random_state=42)

    # Renombramos nuestras columnas para tenerlas en el formato que requiere datasets
    train = big_train[['clean_tweet','hard_label_t1']].rename(columns={'clean_tweet':'text', 'hard_label_t1':'label'})
    valid = valid[['clean_tweet','hard_label_t1']].rename(columns={'clean_tweet':'text', 'hard_label_t1':'label'})
    test = test[['clean_tweet','hard_label_t1']].rename(columns={'clean_tweet':'text', 'hard_label_t1':'label'})
    
    # Convertimos los df en objetos tipo Dataset
    train = Dataset.from_pandas(train)
    test = Dataset.from_pandas(test)
    valid = Dataset.from_pandas(valid)

    return train, valid, test

def preprocess_function(tokenizer):
    """
    Devuelve una función de preprocesamiento que tokeniza los textos del dataset.
    
    Parámetros:
    tokenizer -- El tokenizador de Hugging Face.
    
    Retorna:
    Una función que recibe ejemplos del dataset y los tokeniza.
    """
    def tokenize(examples):
        return tokenizer(examples["text"])
    return tokenize


def compute_metrics(eval_pred):
    """
    Calcula métricas de evaluación: exactitud (accuracy) y F1.

    Parámetros:
    eval_pred -- Una tupla (logits, labels) con las predicciones y etiquetas reales.

    Retorna:
    Un diccionario con los valores de exactitud y F1.
    """
    # Cargar las métricas de Hugging Face
    accuracy_metric = load("accuracy")
    f1_metric = load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Seleccionar la clase más probable

    # Calcular exactitud y F1
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

def timestamp(fmt='%y%m%dT%H%M%S'):
    """
    Genera una marca de tiempo en el formato especificado.

    Parámetros:
    fmt -- El formato de la marca de tiempo (por defecto: 'yymmddTHHMMSS').

    Retorna:
    Una cadena de texto con la marca de tiempo actual.
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)

def tokenizer(tokenizer_name, train_dataset , dev_dataset):
    # Inicializar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenizar los datasets de entrenamiento y evaluacion
    tokenized_train = train_dataset.map(preprocess_function(tokenizer), batched=True)
    tokenized_dev= dev_dataset.map(preprocess_function(tokenizer), batched=True)
    
    print(f"Tokenizacion del modelo terminada")

    return tokenized_train, tokenized_dev,tokenizer


def newTrainingArguments(run_dir, 
                         learning_rate=2e-5, 
                         per_device_train_batch_size=4,
                         per_device_eval_batch_size=4,
                         num_train_epochs=3, weight_decay=0.0001,
                         logging_strategy="epoch", evaluation_strategy="epoch",
                         save_strategy="epoch", load_best_model_at_end=True,
                         no_cuda=False):
    
        # Definir los argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir = run_dir,
            learning_rate = learning_rate,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            num_train_epochs = num_train_epochs ,
            weight_decay = weight_decay,
            logging_strategy = logging_strategy,  # Registrar los logs al final de cada época
            evaluation_strategy = evaluation_strategy,  # Evaluar el modelo al final de cada época
            save_strategy = save_strategy,  # Guardar el modelo al final de cada época
            load_best_model_at_end = load_best_model_at_end,  # Cargar el mejor modelo al final del entrenamiento
            no_cuda = no_cuda, 
            logging_dir=f"{run_dir}/logs"
        )

        return training_args


def save_info(model,
              run_dir, 
              trainer, 
              training_args, 
              tokenizer):
    
    # Guardar informacion de modelo en archivos
    file = open(run_dir+"/param.txt", "w")
    file2 = open(run_dir+"/metrics_val.txt", "w")
    a = file.write(str(training_args))
    b = file2.write(str(trainer.evaluate()))
    file.close()
    file2.close()

    # Guardar los datos del mejor modelo
    model_path = "/bestModel/" 
    model.save_pretrained(run_dir+model_path)
    tokenizer.save_pretrained(run_dir+model_path)

def manual_evaluation(model_dir: str, 
                      language: str = "es", 
                      examples: list = None):
    """
    Evalúa manualmente un modelo de clasificación de texto utilizando ejemplos predefinidos o personalizados.

    Parameters
    ----------
    model_dir : str
        Ruta del directorio del modelo preentrenado.
    language : str, optional
        Idioma de los ejemplos predefinidos. Puede ser "es" (español) o "en" (inglés). 
        Por defecto es "es".
    examples : list, optional
        Lista de frases a evaluar. Si es None, se usarán ejemplos predefinidos según el idioma seleccionado.

    Returns
    -------
    None
        Imprime cada ejemplo junto con su predicción correspondiente.
    """
    # Ejemplos predefinidos según el idioma
    default_examples = {
        "es": ["Maten mujeres", "esta pelicula apesta","esta pelicula es muy buena", "lanzas como niña", "pinche feminista", "las mujeres trans no son mujeres"],
        "en": ["Kill women", "this movie sucks","that is great", "you throw like a girl", "damn feminist", "trans women are not women"]
    }
    
    if examples is None:
        examples = default_examples.get(language, default_examples["es"])
    
    model = pipeline("text-classification", model_dir)
    results = model(examples)

    # Mostrar resultados con el formato "ejemplo --> predicción"
    for example, result in zip(examples, results):
        print(f"{example} --> {result}")
    
    print()



def evaluate_model(test : Dataset ,model_name : str, model_dir : str):
    """
    Evalúa un modelo de clasificación de texto usando un Dataset de Hugging Face.
    
    Args:
        test (Dataset): Dataset de prueba de Hugging Face.
        model_name (str): Nombre del modelo.
        run_dir (str): Directorio donde se encuentra el modelo.
        model_path (str): Ruta del modelo dentro del directorio.
    """

    # Crear el modelo de clasificación de texto utilizando pipeline de Hugging Face
    sentiment_model = pipeline("text-classification", model=model_dir)

    # Obtener las etiquetas de clase del modelo
    model_config = sentiment_model.model.config
    id2label = model_config.id2label  # Diccionario {0: 'not sexist', 1: 'sexist'}

    # Determinar cuál es la etiqueta negativa
    possible_labels = list(id2label.values())
    negative_label = possible_labels[0]  # Asumiendo que la primera es la negativa

    print(f"Etiquetas del modelo: {id2label}")
    print(f"Etiqueta negativa detectada: {negative_label}")

    # Obtener los datos de prueba
    data = test['text']
    predictions = sentiment_model(data)

    # Convertir las predicciones a etiquetas y calcular las métricas
    predicted_labels = np.array([prediction['label'] for prediction in predictions])
    print(f"Predecidas OG: \n {predicted_labels[:10]}")

    predicted_labels = np.where(predicted_labels == negative_label, 0, 1)  # Mapear etiquetas
    print(f"Predecidas Trans: {predicted_labels[:10]}")

    true_labels = test['label']
    print(f"Verdaderas: {true_labels[:10]}")


    # Calcular las métricas
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Imprimir el reporte de clasificación y la matriz de confusión
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))
    print("Confusion Matrix:")
    print(conf_matrix)

    # Crear la carpeta de evaluaciones si no existe
    evaluations_dir = './evaluaciones'
    os.makedirs(evaluations_dir, exist_ok=True)

    # Definir la ruta del archivo de resultados
    results_file_path = join(evaluations_dir, f'{model_name.replace("/", "_")}_evaluation.txt')

    # Guardar las métricas y la matriz de confusión en un archivo de texto
    with open(results_file_path, 'w') as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")
        file.write("Confusion Matrix:\n")
        file.write(str(conf_matrix))
        file.write("\n\nClassification Report:\n")
        file.write(classification_report(true_labels, predicted_labels))

