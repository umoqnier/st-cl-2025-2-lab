# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: tf3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 4: Modelos del Lenguaje Neuronales
#
# Castañeda Mora Carlos
#
# **Fecha de entrega: 6 de abril de 2025 11:59pm**
#
# A partir del modelo entrenado:
#
# - Sacar los embeddings de las palabras del vocabulario
#
# - Visualizar en 2D los embeddings de algunas palabras (quizá las más frecuentes, excluyendo stopwords)
#
# - Seleccione algunas palabras y verifique sí realmente codifican nociones semánticas, e,g, similitud semántica con similitud coseno entre dos vectores, analogías por medios de operaciones de vectores
#
#

# %%

# %% [markdown]
# ### Extra (0.5 pts):
#
# - Correr el modelo de Bengio pero aplicando una técnica de subword tokenization al corpus y hacer generación del lenguaje
#
# * La generación del lenguaje debe ser secuencias de palabras (no subwords)

# %%
