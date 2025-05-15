# -*- coding: utf-8 -*-
"""Practica 6.ipynb
"""

from datasets import load_dataset, Dataset, ClassLabel
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import torch
from codecarbon import EmissionsTracker
from datasets import DatasetDict

tracker_training = EmissionsTracker()
tracker_training.start()

dataset	= load_dataset("wykonos/movies")
ds = dataset["train"].select_columns(["overview", "genres"]).to_pandas()

labels = pd.Series([i for l in ds["genres"].str.split("-") if l is not None for i in l]).unique()
mapper = {j: i for i, j in enumerate(labels)}
with open("labels.json", "w") as f:
    json.dump(mapper, f)
ds = ds.dropna()
ds["genres"] = ds["genres"].str.split("-").apply(lambda x: [mapper[i] for i in x])
ds["genres"] = ds["genres"].apply(lambda x: [1. if i in x else 0. for i in range(len(labels))])
ds.rename(columns={"overview": "text", "genres": "labels"}, inplace=True)
ds

# Dividir en train/validation/test (80%-10%-10%)
ds = ds.sample(frac=1/3, random_state=42)  # Mezclar
train_df = ds[:int(0.75*len(ds))]
val_df = ds[int(0.75*len(ds)):int(0.9*len(ds))]
test_df = ds[int(0.9*len(ds)):]

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False)
})

dataset

model_name = "microsoft/deberta-v3-small"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = DebertaV2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(mapper),  # Ajusta al número de géneros
    problem_type="multi_label_classification"
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    max_length=512,  # Máximo de DeBERTa
)

# Tokenización (igual que antes)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

tokenized_data = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

cfl_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = sigmoid(logits)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return cfl_metrics.compute(
        predictions=predictions,
        references=labels.astype(int).reshape(-1)
    )

# Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./deberta-results",
    per_device_train_batch_size=12,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()

# Evaluación final en el conjunto de test (opcional)
test_results = trainer.evaluate(tokenized_data["test"])
print("Resultados en Test:", test_results)

model.save_pretrained("./deberta-results/final_model")
tokenizer.save_pretrained("./deberta-results/final_tokenizer")
trainer.save_model("./deberta-results/final_trainer")

emissions = tracker_training.stop()
print(f"Emisiones de CO2 durante el entrenamiento: {emissions} kg")

with open("./deberta-results/checkpoint-63546/trainer_state.json", "r") as f:
    trainer_state = json.load(f)

loss_history = pd.DataFrame(trainer_state['log_history'])

plt.plot(loss_history['step'], loss_history['loss'], label='Training Loss')
plt.title("Loss History")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

inference_tracker = EmissionsTracker()
inference_tracker.start()

r_model = DebertaV2ForSequenceClassification.from_pretrained("./deberta-results/final_model")
r_model.to("cuda")
r_tokenizer = DebertaV2Tokenizer.from_pretrained("./deberta-results/final_tokenizer")
with open("labels.json", "r") as f:
    mapper = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_genres(text, threshold=0.5):
    # Tokenizar el texto de entrada
    inputs = r_tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    # Pasar por el modelo
    with torch.no_grad():
        outputs = r_model(**inputs)

    # Convertir logits a probabilidades con sigmoid (para multi-etiqueta)
    probs = torch.sigmoid(outputs.logits)

    # Aplicar umbral para obtener predicciones binarias
    predictions = (probs > threshold).int().squeeze().tolist()

    # Mapear índices a nombres de géneros (ajusta según tu mapeo)
    genre_mapping = {v: k for k, v in mapper.items()}
    # Obtener géneros predichos
    predicted_genres = [genre_mapping[i] for i, pred in enumerate(predictions) if pred == 1]

    return predicted_genres, probs.cpu().numpy().tolist()

# Ejemplo de uso
text = "A catholic priest is sent to a remote island to investigate the disappearance of a young priest. He finds the island is home to a cult that worships a demon."
predicted_genres, probabilities = predict_genres(text)
print(f"Predicted Genres: {predicted_genres}")

emissions = inference_tracker.stop()
print(f"Emisiones de CO2 durante la inferencia: {emissions} kg")