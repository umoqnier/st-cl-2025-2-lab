# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="O7w9L21gsL8B"
# # Práctica 6: *Fine-tuning en producción*
#
# ##**Fecha de entrega: 11 de Mayo de 2025**
#
# ### Cesar Cossio Guerrero

# + [markdown] id="3U-das8lsPTp"
# - Selecciona un modelo pre-entrenado como base y realiza *fine-tuning* para resolver alguna tarea de NLP que te parezca reelevante
#   - Procura utilizar datasets pequeños para que sea viable
#   - Recuerda las posibles tareas disponibles en HF `*For<task>`
# - Desarrolla y pon en producción un prototipo del modelo
#   - Incluye una URL pública donde podamos ver tu proyecto
#   - Recomendamos usar framewoks de prototipado (*streamlit* o *gradio*) y el *free-tier* de *spaces* de hugging face
#     - https://huggingface.co/spaces/launch
#     - https://huggingface.co/docs/hub/spaces-sdks-streamlit
#     - https://huggingface.co/docs/hub/spaces-sdks-gradio
# - Reporta que tan bien se resolvió la tarea y que tan útil fue tu app
# - Reporta retos y dificultades al realizar el *fine-tuning* y al poner tu modelo en producción
#
# ## Extra
#
# - Utiliza [code carbon](https://codecarbon.io/#howitwork) para reportar las emisiones de tu app

# + [markdown] id="fjZDFxO5zSwR"
# # Decidí hacer un **finetunning** en el modelo **Bert** para **clasificación** en un **análisis de sentimientos**

# + [markdown] id="oy6zIfzuy_D5"
# ## Cargamos las **librerías**

# + id="ciXZ-2ccDF9v" jupyter={"outputs_hidden": true}
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# + [markdown] id="kVwOSh26zGQx"
# ## Cargamos el **dataset** que usaremos para hacer el **FineTunning**

# + [markdown] id="ONs90477zuXi"
# ### Si **falla** la **carga** del **dataset** por favor usar el siguiente comando que actuliza la libreríá datasets

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="8mkuXsS6uvhQ" outputId="eb0ab646-6712-48a8-dd9c-c81cef5f21c0"
#pip install -U datasets   # En caso de que fallé la carga del dataset de la librería datasets

# + id="MPu8hLroSQcd"
ds = load_dataset("mteb/tweet_sentiment_extraction")

# + colab={"base_uri": "https://localhost:8080/"} id="ReoZ3wB5_we_" outputId="9fbad45a-e9ad-4df2-e664-0fcee5543c75"
ds

# + [markdown] id="L5CLwM0RALmH"
# ### Hay 3 categorías, solo usaré dos para que el entrenamiento sea más rápido

# + id="mygnXIq--tfa"
ds_filtered = ds.filter(lambda example: example['label'] in [0, 1])

# + [markdown] id="2P9pVwHZAVO3"
# ### Tomaré solo el **1%** del total del dataset

# + colab={"base_uri": "https://localhost:8080/"} id="_vEyhbp9-6lP" outputId="03cc5027-e8fd-43c1-d872-558c8efa46b5"
from sklearn.model_selection import train_test_split

# Convertir el dataset de Hugging Face a un DataFrame de pandas
ds_filtered_df = pd.DataFrame(ds_filtered['train'])

# Tomar una muestra del 10% de manera estratificada
ds_sample, _ = train_test_split(
    ds_filtered_df,
    train_size=0.01,
    stratify=ds_filtered_df['label'],
    random_state=42 # Para reproducibilidad
)

print("Shape of the original dataset:", ds_filtered_df.shape)
print("Shape of the sampled dataset:", ds_sample.shape)
print("\nValue counts in the original dataset label column:")
print(ds_filtered_df['label'].value_counts(normalize=True))
print("\nValue counts in the sampled dataset label column:")
print(ds_sample['label'].value_counts(normalize=True))

# + [markdown] id="vTSY3j9jAcv6"
# ### Lo regreso a su forma original, DatasetDict para entrenar al modelo

# + colab={"base_uri": "https://localhost:8080/"} id="ZKhr9BDb_mNS" outputId="2096133e-1b45-48d8-d60e-d3bd876c2338"
from datasets import Dataset, DatasetDict

# Convertir el DataFrame de pandas de vuelta a un objeto Dataset de Hugging Face
ds_sample = Dataset.from_pandas(ds_sample)

# Crear un DatasetDict (aunque solo tengamos el split 'train')
dataset = DatasetDict({'train': ds_sample})

dataset

# + id="B120_W2PVIr1"
train_df =dataset['train'].to_pandas()

# + colab={"base_uri": "https://localhost:8080/", "height": 178} id="LI9Nuf5F1eyc" outputId="fc78e887-c713-47d6-fd53-8807162a4d0a"
train_df.value_counts('label')

# + [markdown] id="sHLnjSOD1IF6"
# ### Usaremos solo **dos categorías** del conjunto para mejorar el **desempeño**: **positivo** y **negativo**

# + colab={"base_uri": "https://localhost:8080/", "height": 178} id="IHyTPu_joGW_" outputId="edf96cf5-b058-458a-ee95-5d74d3965c06"
train_df.value_counts('label')

# + id="AcSoNK2abEag"
pad_len = 13

# + [markdown] id="7QVkRlZ41l5T"
# ### Creamos los conjuntos de entre

# + id="SBvEiorDDZSE"
#load model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# + [markdown] id="AxJp1gLpAoTv"
# ### Se realiza la **tokenización** para entrenar al modelo

# + colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["f5958ed71f494d65afefc6ff400a5ecc", "9cd5248f99104c03a5db56c9ac2accb6", "1eefb6e43afe43d7bb3051fbafef483e", "8725d9e8db1a4792b211d3fadb041df2", "4c4865afc7a343cab53281d0f60208dc", "f31c3a3fece641bf8fc2fd7f4e587dfe", "918683fd2127458a9a2b67f4b444df65", "c4499429206f4aa997fec01f05efc307", "65ed6e89ec6443d89cf20a69d8546044", "4fe84168402545ceb032a08eef3838c4", "cfef475853324373837e49a58a2e0bbf"]} id="iUc1OLox9e0b" outputId="f5af9c4c-c619-4394-efae-09d5531890a9"
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# + [markdown] id="ONZQ6c63BZVH"
# ### Se crea potencialmente un conjunto de entrenamiento, de prueba y validación. No los usé por motivos de hardware.

# + id="A0muOOtD9qZ7"
train_testvalid = tokenized_datasets['train'].train_test_split(test_size=0.2)
train_dataset = train_testvalid['train']
valid_dataset = train_testvalid['test']

# + colab={"base_uri": "https://localhost:8080/"} id="66HBisWiGOCt" outputId="91d897dc-b4aa-4fa8-82c4-9cb30c14c381"
train_dataset.shape

# + id="NbJaUasg-WWE"
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)

# + id="woddhjzfAn7E"
from torch.optim import AdamW

# + [markdown] id="g6WvwjXrBkq3"
# ### Inicializamos el modelo **Bert uncased**

# + colab={"base_uri": "https://localhost:8080/"} id="nefhsHCf-ZZM" outputId="89b44556-2d87-4faf-8090-f94847142cef"
from transformers import BertForSequenceClassification #, AdamW

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# + id="LqxRCiSaHF0t"
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)

# + [markdown] id="UyG4wXzzBq5_"
# Definimos **parámetros** del entrenamiento para el **finetunnig**

# + colab={"base_uri": "https://localhost:8080/"} id="Gb_TeaVR-iu0" outputId="b2687e15-c703-4f39-f721-240313814e6c"
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    report_to=["tensorboard"]
    #learning_rate=2e-5,
    #per_device_train_batch_size=8,
    #per_device_eval_batch_size=8,
    #num_train_epochs=3,
    #weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    # Esto es nuevo O:
    #compute_metrics=compute_metrics,
)

# + [markdown] id="-naVSjdcB6k3"
# ### Se **entrena** el modelo

# + colab={"base_uri": "https://localhost:8080/", "height": 221} id="9z3luaPpBojd" outputId="675cbd7a-c62c-4581-c123-5b43926dd991"
trainer.train()

# + [markdown] id="nYxebodzB-Nw"
# ### **Salvamos el modelo**

# + id="3Qw3qfEu9ZLg"
trainer.save_model("./my_bert_model")

# + [markdown] id="ecI4BnJQCBIH"
# ### Aquí podemos **probar** el **modelo**

# + colab={"base_uri": "https://localhost:8080/"} id="KV10qSez-NbW" outputId="a1fee0d7-a9e2-4b7e-ac96-8cd47d9785ba"
# Define the sentence you want to predict
sentence = "I am very sad"

# Tokenize the sentence
tokens_sentence = tokenizer(
    sentence,
    max_length=13,
    padding='max_length',
    truncation=True,
    return_tensors='pt'  # Return PyTorch tensors
)

# Prepare tensors for the model and move to the correct device
# Use 'cpu' if you are not using a GPU, otherwise use 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = tokens_sentence['input_ids'].to(device)
attention_mask = tokens_sentence['attention_mask'].to(device)

# Get predictions using the loaded model
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    preds = model(input_ids, attention_mask)

# Convert predictions (logits) to probabilities (if needed) and then to class labels
# The model outputs raw logits, so we need to find the index of the highest logit
predicted_class_index = torch.argmax(preds.logits, dim=1).cpu().numpy()[0]

# Map the predicted class index back to a meaningful label
# Assuming your model is trained for binary classification with labels 0 and 1
# You need to know what these labels represent in your specific model
sentiment_map = {0: 'negative', 1: 'positive'} # Adjust this based on your training
predicted_sentiment = sentiment_map.get(predicted_class_index, 'unknown')

print(f"The sentence '{sentence}' is classified as: {predicted_sentiment}")


# + [markdown] id="q65-IQ2nedrl"
# # **Load Bert Architecture**

# + colab={"base_uri": "https://localhost:8080/"} id="93ApCh9ECXWC" outputId="17db06a9-85a0-4447-a225-2ea8433bc207"
# **Load the fine-tuned model**
loaded_model = BertForSequenceClassification.from_pretrained("./my_bert_model")
loaded_tokenizer = BertTokenizerFast.from_pretrained("./my_bert_model")

# Move the loaded model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

# Example usage with the loaded model (similar to the testing part)
sentence = "Today is horrible!"

# Tokenize the sentence using the loaded tokenizer
tokens_sentence = loaded_tokenizer(
    sentence,
    max_length=13,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Prepare tensors for the loaded model and move to the correct device
input_ids = tokens_sentence['input_ids'].to(device)
attention_mask = tokens_sentence['attention_mask'].to(device)

# Get predictions using the loaded model
loaded_model.eval()  # Set model to evaluation mode
with torch.no_grad():
    preds = loaded_model(input_ids, attention_mask)

# Convert predictions (logits) to probabilities and then to class labels
predicted_class_index = torch.argmax(preds.logits, dim=1).cpu().numpy()[0]
sentiment_map = {0: 'negative', 1: 'positive'} # Adjust this based on your training
predicted_sentiment = sentiment_map.get(predicted_class_index, 'unknown')

print(f"Using the loaded model, the sentence '{sentence}' is classified as: {predicted_sentiment}")

# + [markdown] id="yyZUMednGWiZ"
# ### Esta práctica se me hizo muy interesante, pero por más la más difícil. Me costó mucho encontrar un buen lugar donde correr el fine tuning. También escojer la tarea y el dataset se me hizo un poco complicado porque no sabía muy bien como podría entrenarlo. Finalmente busqué un tutorial en HuggingFace.
#
# ### Creo que es más complicado de lo que parece. Crear la app no fue difícil, realmente creo que es sencillo y muy poderoso.
#
# ### No pude entrenar muy bien el modelo por problemas de hardware pues tardaba mucho tiempo, pero me parece que aprendía bastante.

# + id="oR9IbJjBCk-5"

