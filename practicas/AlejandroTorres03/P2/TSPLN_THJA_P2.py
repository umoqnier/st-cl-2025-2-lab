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
#     name: python3
# ---

# %%
# !pip install spacy datasets matplotlib
# !python -m spacy download es_core_news_md

# %% [markdown] id="ZPUhoPghmtFC"
# ## Práctica 2: Propiedades estadísticas de la lengua
#
# ### Fecha de entrega: 2 de Marzo de 2025 11:59pm
#
# 1. Verificar si la ley de Zipf se cumple en un lenguaje artificial creado por ustedes.
#     - *Ejemplo:* Un "lenguaje artificial" podría ser simplemente un texto donde las secuencias de caracteres fueron generadas aleatoriamente.
# 2. Explorar `datasets` del sitio [Hugging Face](https://huggingface.co/datasets) y elegir documentos de diferentes dominios en Español (al menos 3). Realizar reconocimiento de entidades nombradas (NER).
#     - Pueden utilizar subconjuntos de los datasets encontrados
#     - Mostrar resultados del reconocimiento
#     - Una distribución de frecuencias de las etiquetas más comunes en cada dominio
#     - Comentarios generales del desempeño observado.
#
# *Sugerencias: Spacy, CoreNLP (puede ser cualquier otra herramienta)*

# %% [markdown]
# ## 1. Ley de Zipf

# %%
import random
import string
from collections import Counter
import matplotlib.pyplot as plt

# --- Configuración ---
num_total_palabras = 10000  # Total de palabras en el texto
num_palabras_comunes = 20   # Número de palabras frecuentes (ej: "el", "la", "de")
num_palabras_raras = 500     # Número de palabras raras (aparecerán poco)

# --- Generar palabras ---
# 1. Palabras "comunes" (aparecerán mucho)
palabras_comunes = [''.join(random.choices(string.ascii_lowercase, k=3)) for _ in range(num_palabras_comunes)]

# 2. Palabras "raras" (aparecerán poco)
palabras_raras = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(num_palabras_raras)]

# --- Asignar frecuencias ---
# - Las palabras comunes aparecerán entre 100 y 500 veces.
# - Las palabras raras aparecerán entre 1 y 10 veces.
texto_artificial = []

# Añadir palabras comunes (muchas repeticiones)
for palabra in palabras_comunes:
    frecuencia = random.randint(100, 500)
    texto_artificial.extend([palabra] * frecuencia)

# Añadir palabras raras (pocas repeticiones)
for palabra in palabras_raras:
    frecuencia = random.randint(1, 10)
    texto_artificial.extend([palabra] * frecuencia)

# Mezclar todas las palabras para que no estén ordenadas
random.shuffle(texto_artificial)

# Recortar a num_total_palabras (por si nos pasamos)
texto_artificial = texto_artificial[:num_total_palabras]

# --- Calcular frecuencias ---
word_counts = Counter(texto_artificial)
sorted_counts = sorted(word_counts.values(), reverse=True)
ranks = range(1, len(sorted_counts) + 1)

# --- Graficar (log-log) ---
plt.figure(figsize=(10, 6))
plt.loglog(ranks, sorted_counts, 'o', markersize=5, alpha=0.7)
plt.xlabel('Rango (log)')
plt.ylabel('Frecuencia (log)')
plt.title('Ley de Zipf en Lenguaje Artificial')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Análisis
#
# El lenguaje artificial generado muestra un comportamiento parcialmente similar a la Ley de Zipf, especialmente en los rangos altos (palabras más frecuentes). No obstante, la presencia artificial y uniforme de muchas palabras raras altera la distribución esperada, lo que demuestra que la ley de Zipf no se cumple completamente en un lenguaje no natural.

# %% [markdown]
# ## 2. NER

# %%
from datasets import load_dataset

# Dataset 1: Jurídico Mexicano
dataset_juridico = load_dataset("Danielbrdz/Barcenas-Juridico-Mexicano-Dataset", split="train[:500]")  # Subconjunto para prueba

# Dataset 2: Narrativas (LAMBADA en español)
dataset_lambada = load_dataset("EleutherAI/lambada_openai", "es", split="test[:500]")

# Dataset 3: Quran en español
dataset_quran = load_dataset("nazimali/quran", split="train[:500]")

# %%
spacy.load('es_core_news_md')

# %%
# Mira el primer elemento del dataset jurídico
print(dataset_juridico[0])

# Para el dataset LAMBADA
print(dataset_lambada[0])

# Para el Quran
print(dataset_quran[0])

# %%
import spacy
nlp = spacy.load("es_core_news_md")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Ejemplo para un texto del dataset jurídico:
text_juridico = dataset_juridico[0]["question"]  # Ajusta según la estructura real del dataset
entities = extract_entities(text_juridico)
print(entities)

# %%
from collections import defaultdict
import spacy

nlp = spacy.load("es_core_news_md")

def count_entities(dataset, text_field="text"):  # "text" es el valor por defecto, pero lo cambiaremos
    entity_counts = defaultdict(int)
    for example in dataset:
        text = example[text_field]  # Accede al campo correcto (ej: "question" o "translation")
        doc = nlp(text)
        for ent in doc.ents:
            entity_counts[ent.label_] += 1
    return entity_counts


# %%
# Dataset Jurídico (campo "question")
counts_juridico = count_entities(dataset_juridico, text_field="question")

# Dataset Quran (campo "surah")
counts_quran = count_entities(dataset_quran, text_field="translation-es-garcia")

# Dataset LAMBADA (campo "text" por defecto)
counts_lambada = count_entities(dataset_lambada)  

# %%
import pandas as pd

def plot_entity_counts(counts, title):
    df = pd.DataFrame.from_dict(counts, orient="index", columns=["Frecuencia"])
    df.sort_values(by="Frecuencia", ascending=False).plot(kind="bar", title=title)

plot_entity_counts(counts_juridico, "Entidades en Dataset Jurídico (question)")
plot_entity_counts(counts_quran, "Entidades en Dataset Quran (surah)")
plot_entity_counts(counts_lambada, "Entidades en Dataset LAMBADA (text)")

# %% [markdown]
# ### Análisis de Reconocimiento de entidades (NER)
#
# Los resultados muestran diferencias interesantes:
#
# * En el **dataset jurídico**, las entidades más comunes fueron `LOC`, `ORG` y `MISC`, mientras que `PER` (personas) fue casi inexistente. Esto tiene sentido ya que el lenguaje jurídico se enfoca en lugares, instituciones y términos legales más que en individuos.
#
# * En el **dataset religioso (Quran)**, predominan `MISC` y `PER`, lo cual es coherente con la naturaleza del texto, que menciona con frecuencia conceptos abstractos y figuras históricas o religiosas.
#
# * En el **dataset narrativo (LAMBADA)**, `PER` fue la entidad más frecuente, seguida de `MISC` y `LOC`. Esto refleja que los textos narrativos tienden a enfocarse en personajes, lugares y elementos variados del entorno.
#
# En general, el modelo mostró un desempeño razonable para identificar entidades relevantes, aunque en textos complejos o muy específicos (como el jurídico o el religioso) pueden aparecer confusiones o etiquetas genéricas (`MISC`) debido a la ambigüedad o falta de contexto entrenado.
#

# %% [markdown]
#
