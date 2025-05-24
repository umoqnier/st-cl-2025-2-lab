# -*- coding: utf-8 -*-
"""
# ***Dependencias y bibliotecas***

## ***Dependencias***
"""

!pip install codecarbon
!pip install hf_xet

"""## ***Bibliotecas***"""

# Datasets
from datasets import load_dataset
import pandas as pd

# NLP
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Trainer, TrainingArguments

# Math
import numpy as np

# Metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Codecarbon
from codecarbon import EmissionsTracker

# Format
from rich.console import Console
from rich.table import Table

# Kaggle secrets
from kaggle_secrets import UserSecretsClient

# Hugging face interface
from huggingface_hub import HfApi, login, create_repo, upload_folder

"""# ***Carga de datos***

Proporciona ~2k mensajes de chat de partidas de Dota 2 con niveles de toxicidad (0=no tóxico, 1=leve, 2=alto).
"""

ds = load_dataset("dffesalbon/dota-2-toxic-chat-data")
print(ds["train"][0])

"""# ***Cargando el modelo pre-entrenado***"""

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Carga del modelo RoBERTa para clasificación con 3 etiquetas
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=3
)

model.config.id2label = {
    0: "non-toxic",
    1: "mild toxicity",
    2: "toxic",
}

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["message"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokenized["labels"] = examples["target"]
    return tokenized

tokenized_ds = ds.map(tokenize_function, batched=True)

tokenized_ds = tokenized_ds.remove_columns(["message", "target"])
tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

"""# ***Definición de métricas***"""

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),  # F1 promedio
        "report": classification_report(labels, predictions)         # Reporte completo
    }

"""# ***Entrenamiento***"""

training_args = TrainingArguments(
    output_dir="./dota2-toxicity-model",
    eval_strategy="epoch",

    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,

    learning_rate=2e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=7,

    save_strategy="epoch",
    save_total_limit=1,

    logging_dir="./logs",
    report_to="none"
)


tracker = EmissionsTracker(
    log_level="critical",  # Evita logs redundantes
    project_name="dota2_toxicity_classification",
    output_dir="."
)

try:
    # Inicia el tracking
    tracker.start()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()  # ¡Ahora debería funcionar
finally:
    # Detén el tracker y guarda el reporte
    emissions = tracker.stop()

df = pd.read_csv("emissions.csv")

# Extrae un solo registro (el último)
row = df.iloc[-1]

# 3) Compute additional metrics (convert kg to g, kWh to Wh)
emissions_g = row['emissions'] * 1000                      # kg CO2e -> g CO2e
energy_consumed_wh = row['energy_consumed'] * 1000         # kWh -> Wh
cpu_energy_wh = row['cpu_energy'] * 1000                   # kWh -> Wh
gpu_energy_wh = row['gpu_energy'] * 1000                   # kWh -> Wh
ram_energy_wh = row['ram_energy'] * 1000                   # kWh -> Wh

# 4) Build and display the Rich table
table = Table(title="💨 CodeCarbon Emissions & Resource Report")

table.add_column("Metric", style="cyan", no_wrap=True)
table.add_column("Value", style="magenta", justify="right")

table.add_row("Duration (s)", f"{row['duration']:.1f}")
table.add_row("CPU Energy (Wh)", f"{cpu_energy_wh:.2f}")
table.add_row("GPU Energy (Wh)", f"{gpu_energy_wh:.2f}")
table.add_row("RAM Energy (Wh)", f"{ram_energy_wh:.2f}")
table.add_row("Total Energy (Wh)", f"{energy_consumed_wh:.2f}")
table.add_row("Emissions (g CO₂e)", f"{emissions_g:.2f}")
table.add_row("Emissions Rate (kg CO₂e/kWh)", f"{row['emissions_rate']:.4f}")
table.add_row("Tracking Mode", row['tracking_mode'])
table.add_row("CPU Model", row['cpu_model'])
table.add_row("CPU Count", str(int(row['cpu_count'])))
table.add_row("GPU Model", row['gpu_model'] or "N/A")
table.add_row("GPU Count", str(int(row['gpu_count'])))
table.add_row("RAM Total (GB)", f"{row['ram_total_size'] / 1024:.2f}")
table.add_row("Region", row['region'])
table.add_row("PUE", f"{row['pue']:.2f}")

console = Console()
console.print(table)

"""# ***Evaluación***"""

# Evaluación final:
metrics = trainer.evaluate(tokenized_ds["test"])
console = Console()
table = Table(title="📊 Métricas de Evaluación - Toxicidad en Dota 2", show_header=True, header_style="bold magenta")

# Columnas
table.add_column("Métrica", style="cyan", width=20)
table.add_column("Valor", style="green", justify="right")

# Agregar filas
table.add_row("Pérdida (Loss)", f"{metrics['eval_loss']:.4f}")
table.add_row("Accuracy", f"{metrics['eval_accuracy']:.2%}")
table.add_row("F1 Macro", f"{metrics['eval_f1_macro']:.2%}")

# Imprimir tabla
console.print(table)

# Imprimir el reporte de clasificación con formato
console.print("\n[bold]📝 Reporte de Clasificación:[/bold]")
console.print(metrics["eval_report"])

"""# ***Aplicación***"""

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    function_to_apply="softmax",  # Para probabilidades
)

chat_message = "You're a noob, uninstall the game!"

# Predecir
pred = classifier(chat_message)

table = Table(title=f'Resultados de Clasificación para "{chat_message}"')
table.add_column("Etiqueta", style="cyan", justify="left")
table.add_column("Probabilidad", style="magenta", justify="right")

for entry in pred:
    label = entry["label"]
    score = entry["score"]
    # Resalta la etiqueta con mayor probabilidad
    if entry == max(pred, key=lambda x: x["score"]):
        table.add_row(f"[bold]{label}[/bold]", f"[bold]{score:.4f}[/bold]")
    else:
        table.add_row(label, f"{score:.2f}")

# Mostrar tabla
console = Console()
console.print(table)

"""# ***App in Huggin Face***

## ***Guardamos el modelo***
"""

trainer.save_model("dota2-toxic-detector")

"""## ***Subimos el modelo a HF***"""

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_PLN")  # Usa el nombre de tu secreto
login(token=hf_token)

!huggingface-cli login --token {hf_token}

model.push_to_hub("fwgalde/dota2-toxic-detector")
tokenizer.push_to_hub("fwgalde/dota2-toxic-detector")

api = HfApi()
api.upload_folder(
  folder_path="dota2-toxic-detector",
  repo_id="fwgalde/dota2-toxic-detector",
  repo_type="model",
  create_pr=True
)