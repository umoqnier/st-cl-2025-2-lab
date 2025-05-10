# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Práctica 6: *Fine-tuning en producción*
#
# **Fecha de entrega: 11 de Mayo de 2025**

# +
# Bibliotecas
import pandas as pd
import numpy as np
from os.path import join
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizerñ
from transformers import DataCollatorWithPadding
import torch
import os

import utils


# Descargadas
from codecarbon import EmissionsTracker
import gradio as gr
# -

# ## Hacer el fine tuning del modelo

#
# - Selecciona un modelo pre-entrenado como base y realiza *fine-tuning* para resolver alguna tarea de NLP que te parezca reelevante
#   - Procura utilizar datasets pequeños para que sea viable
#   - Recuerda las posibles tareas disponibles en HF `*For<task>`

# Detectar el dispositivo (CPU o GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# +
# rutas con los datos
main_dir = "./data"
train_rute = main_dir + '/train_es_LimpiezaFinal.csv'
test_rute = main_dir + '/dev_es_LimpiezaFinal.csv'

# Nombre del modelo
model_name = "delarosajav95/HateSpeech-BETO-cased-v2"
# Nombre del tokenizador
tokenizer_name = "delarosajav95/HateSpeech-BETO-cased-v2"

# Definir el directorio de salida con una marca de tiempo
run_dir = join('./Ejecuciones', 'runs', f'Model_{model_name.replace("/", "_")}', utils.timestamp())

# Cargar los datos y procesarlos
train_dataset,eval_dataset,test_dataset = utils.prepareDatasets(train_rute, test_rute)

# Tokenizar los datos 
tokenized_train, tokenized_dev, tokenizer = utils.tokenizer(tokenizer_name,
                                                                train_dataset,
                                                                eval_dataset)

# Preparar el objeto para la recolección y padding dinámico de los datos
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Inicializar el modelo pre-entrenado
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Generar los argumentos de entrenamiento
training_args= utils.newTrainingArguments(run_dir)

# Configurar el entrenador (Trainer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics= utils.compute_metrics
)

# Iniciar el rastreador
tracker = EmissionsTracker()
tracker.start()

# Entrenar el modelo
trainer.train()
print(f"Entrenamiento del modelo {model_name} terminada")

# Evaluar el modelo en el conjunto de prueba
evaluation_results = trainer.evaluate()

print(f"Evaluacion del modelo {model_name} terminada")
print(evaluation_results)

# Guardar la informacion del entrenamiento y la evaluacion
utils.save_info(model,run_dir,trainer,training_args,tokenizer)

# Detener y mostrar emisiones
emissions = tracker.stop()
print(f"Emisiones estimadas durante el ajuste fino del modelo: {emissions:.6f} kg CO₂")
# -

# ## Poner en producción el modelo

# Este codigo se encuentra en la carpeta *detectorSexismo*
#
# El url de la aplicación es : https://huggingface.co/spaces/diana-salgado/detectorSexismo 

# ## Reporte de la actividad

# Durante esta actividad, desarrollé y publiqué una aplicación web basada en un modelo de lenguaje entrenado para clasificar textos en español como sexistas o no sexistas. La tarea se resolvió de forma satisfactoria, ya que el modelo logra ofrecer resultados interpretables y rápidos para cualquier oración corta escrita en español, lo cual resulta útil tanto en contextos académicos como en análisis social o monitoreo de contenido en redes.
#
# **Utilidad de la Aplicación**
#
# La app es particularmente útil para usuarios que desean identificar sesgos de género en lenguaje cotidiano, especialmente en publicaciones de redes sociales como Twitter. Al estar disponible públicamente en Hugging Face Spaces, es accesible desde cualquier dispositivo y puede servir como herramienta educativa o de apoyo en proyectos de investigación relacionados con el análisis de discurso o igualdad de género.
#
# **Fine-tuning del Modelo**
#
# El entrenamiento del modelo fue relativamente sencillo, ya que reutilicé código previamente desarrollado durante mi servicio social. Originalmente, probé múltiples modelos antes de seleccionar el más efectivo para español: [delarosajav95/HateSpeech-BETO-cased-v2]. El conjunto de datos utilizado fue el corpus de la competencia EXIST 2024, compuesto por tweets en español, al cual se le aplicó preprocesamiento (limpieza de símbolos no alfanuméricos, stopwords, conversión a minúsculas, etc.).
#
# Dado que entrenar en mi laptop con CPU resultaba demasiado lento, realicé el ajuste fino en una computadora del laboratorio LATTE del instituto, equipada con una NVIDIA GeForce RTX 3080, logrando completar el entrenamiento en menos de 5 minutos.
#
# **Producción y Despliegue**
#
# Poner el modelo en producción representó un mayor reto, ya que era la primera vez que lo publicaba en Hugging Face Spaces. Aunque inicialmente fue una curva de aprendizaje considerable, el proceso fue bastante amigable gracias a la documentación oficial de Hugging Face y al apoyo recibido de herramientas como ChatGPT. Aprendí a empaquetar correctamente el modelo, generar el archivo requirements.txt, y estructurar el código para que fuese compatible con Gradio.

# ## Extra

# **Reporte de emisiones con CodeCarbon**
#
# Utilicé CodeCarbon para medir el impacto ambiental del ajuste fino de mi modelo. El entrenamiento fue realizado en una máquina con GPU NVIDIA GeForce RTX 3080, y las emisiones estimadas fueron de **0.0041 kg de CO₂**, lo que equivale aproximadamente a la energía consumida por una lámpara LED encendida durante unas 3 horas. No fue posible medir las emisiones de la aplicación en producción, ya que Hugging Face Spaces no permite el monitoreo del consumo energético del entorno, por lo que CodeCarbon no puede ser utilizado de forma efectiva en ese contexto.
