# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 6: *Fine-tuning en producción*

# %% [markdown]
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

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import numpy as np



# %%
# Dataset pequeño: clasificación de sentimientos (positivo / negativo)
dataset = load_dataset("glue", "sst2")

# Tokenizador y modelo base
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)


# %%
# Tokenización de cada oración
def tokenize(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Cargamos métrica estándar de SST-2 (accuracy)
metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



# %%
training_args = TrainingArguments(
    output_dir="sentiment-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    report_to="none"
)


# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(2000)),  # Subset de 2000 muestras
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()


# %%
from huggingface_hub import notebook_login
notebook_login()

# %%
model.push_to_hub("fine-tuned-sentiment-v1", commit_message="Agregué nombres legibles a las etiquetas")
tokenizer.push_to_hub("fine-tuned-sentiment-v1")

