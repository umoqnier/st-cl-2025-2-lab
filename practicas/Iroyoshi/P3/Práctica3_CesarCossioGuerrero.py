# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="qVc146F4b0XG"
# # 3. Práctica: Vectores a palabras
#
# **Fecha de entrega: 16 de Marzo de 2025 @ 11:59pm**
#
# Obtenga la matriz de co-ocurrencia para un corpus en español y realice los siguientes calculos:
# - Las probabilidades conjuntas
# $$p(w_i,w_j) = \frac{c_{i,j}}{\sum_i \sum_j c_{i,j}}$$
# - Las probabilidades marginales
# $$p(w_i) = \sum_j p(w_i,w_j)$$
# - Positive Point Wise Mutual Information (PPMI):
# $$PPMI(w_i,w_j) = \max\{0, \log_2 \frac{p(w_i,w_j)}{p(w_i)p(w_j)}\}$$
#
# **Comparación de representaciones**
#
# Aplica reducción de dimensionalidad (a 2D) de los vectores de la matríz con PPMI y de los vectores entrenados en español:
#
# - Realiza un plot de 100 vectores aleatorios (que esten tanto en la matríz como en los vectores entrenados)
# - Compara los resultados de los plots:
#     - ¿Qué representación dirías que captura mejor relaciones semánticas?
#     - Realiza un cuadro comparativo de ambos métodos con ventajas/desventajas
#
# ### 📁 [Carpeta con vectores](https://drive.google.com/drive/folders/1reor2FGsfOB6m3AvfCE16NOHltAFjuvz?usp=drive_link)

# + [markdown] id="gY61QROzO4Zk"
# ## Cargamos **liberías** que usaremos más adelante
# -

#Reiniciar la sesión después de correr esta celda"
#No es necesario volver a correr esta celda"
# !pip install gensim

import numpy as np

from gensim.models import Word2Vec

import numpy as np
import pandas as pd
import nltk
nltk.download('cess_esp')
nltk.download('stopwords')
from nltk.corpus import cess_esp, stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
from rich import print as rprint
from itertools import combinations, chain
import matplotlib.pyplot as plt

# + [markdown] id="XxY7RW8qjqXa"
# ### Este es el tipo de **oraciones** que hay en la base de datos

# + colab={"base_uri": "https://localhost:8080/", "height": 533} id="v4uYqV8o1bc8" outputId="b67471e3-bf6d-4916-a55c-c531f12e3c86"
# Exploración del corpus
rprint(len(cess_esp.sents()))
rprint(cess_esp.sents()[11])


# + [markdown] id="rF89kbxbR5lM"
# ### Definiremos las mismas funciones vistas en clase para crear la matriz de **co-ocurrencias**.

# + id="HmY_RxvV1cAd"
def get_coocurrence_matrix(sentences: list[list[str]], indices: dict[str, int], n: int = 2) -> np.ndarray:
    matrix = np.zeros((len(indices), len(indices)))
    for sent in sentences:
        for term1, term2 in combinations(sent, 2):
            matrix[indices[term1], indices[term2]] += 1
            matrix[indices[term2], indices[term1]] += 1
    return matrix


# + id="NKo6AIkhAFtv"
def normalize_sent(sentence: list[str]) -> list[str]:
    stemmer = SnowballStemmer("spanish")
    result = []
    return [
        stemmer.stem(word.lower())
        for word in sentence
        if word.isalpha() and
        word.lower() not in stopwords.words("spanish")
    ]

def normalize_corpus(corpus: list[list[str]]) -> list[list[str]]:
    return [
        normalize_sent(sent)
        for sent in corpus
        if len(normalize_sent(sent)) > 1
    ]


# + [markdown] id="K5e6uzbHjylj"
# ### Se crean los elementos básicos para crear la matriz de co-ocurrencias

# + id="C-amJIomBBOI"
sentences = normalize_corpus(cess_esp.sents())

# + id="lUMfOPL7Bh3w"
tokens = list(chain(*sentences))

# + colab={"base_uri": "https://localhost:8080/"} id="ma2rHtFUCsBg" outputId="091e18d4-abea-48b0-a101-9c0718221558"
freqs = Counter(tokens)
freqs.most_common(10)

# + colab={"base_uri": "https://localhost:8080/"} id="idigPDM3CtWg" outputId="28c2cd7b-95c5-4343-c840-cf5dee66f863"
index = {token: idx for idx, token in enumerate(freqs.keys())}
vocabulario = list(index.keys())
vocabulario[0:10]

# + [markdown] id="58uGImtpuC18"
# ## Esta es la matriz de **Co-ocurrencias**

# + colab={"base_uri": "https://localhost:8080/", "height": 443} id="WiDE5ZDICvxY" outputId="1aa85658-7801-479e-b721-908b00dfa4fb"
coocurrence_matrix = get_coocurrence_matrix(sentences, index)
coocurence_df = pd.DataFrame(data=coocurrence_matrix, index=list(index.keys()), columns=list(index.keys()))
coocurence_df

# + [markdown] id="7uOHaFMwkHXT"
# ### Una pequeña vista de la forma en que se organiza la matriz de co-ocurrencias

# + colab={"base_uri": "https://localhost:8080/", "height": 458} id="boibg8Q5CyEY" outputId="4967b2c6-6c78-4f44-ec43-570c89d1aa0a"
query = SnowballStemmer("spanish").stem("presidencial")
coocurence_df[query].sort_values(ascending=False)

# + [markdown] id="t2YsYlx2PFUT"
# ## Calculemos las siguientes **probabilidades**

# + [markdown] id="ehEs4OhiG1l6"
# - **Las probabilidades conjuntas**
# $$p(w_i,w_j) = \frac{c_{i,j}}{\sum_i \sum_j c_{i,j}}$$
#

# + colab={"base_uri": "https://localhost:8080/", "height": 443} id="6w8Fsh0sG1es" outputId="a0007410-2d7c-4829-eb86-027357c669f3"
PROBA_TOTAL = coocurence_df.sum().sum()
prob_conj = coocurence_df.copy()/Total
prob_conj

# + [markdown] id="g7gIaqFOG1Wy"
# - **Las probabilidades marginales**
# $$p(w_i) = \sum_j p(w_i,w_j)$$
#

# + colab={"base_uri": "https://localhost:8080/", "height": 458} id="jE8bOPagG1OT" outputId="7f9db665-eb73-4ab8-ce6c-2613613d2b76"
prob_mut = prob_conj.sum(axis=1).copy()
prob_mut

# + [markdown] id="D7IsBLahG1DC"
# - **Positive Point Wise Mutual Information**
# $$PPMI(w_i,w_j) = \max\{0, \log_2 \frac{p(w_i,w_j)}{p(w_i)p(w_j)}\}$$

# + colab={"base_uri": "https://localhost:8080/", "height": 478} id="lPAlmD2OG04S" outputId="921e718f-7ff9-4f95-c0cf-3fa0eeb0055a"
mut_inf = np.log2(prob_conj/np.outer(prob_mut,prob_mut))
mut_inf[mut_inf < 0] = 0
mut_inf

# + [markdown] id="r6h1pdndN1yM"
# ## **Comparación de representaciones**
#
# Aplica reducción de dimensionalidad (a 2D) de los vectores de la matríz con PPMI y de los vectores entrenados en español:
#
# - Realiza un plot de 100 vectores aleatorios (que esten tanto en la matríz como en los vectores entrenados)
# - Compara los resultados de los plots:
#     - ¿Qué representación dirías que captura mejor relaciones semánticas?
#     - Realiza un cuadro comparativo de ambos métodos con ventajas/desventajas
#
# ### 📁 [Carpeta con vectores](https://drive.google.com/drive/folders/1reor2FGsfOB6m3AvfCE16NOHltAFjuvz?usp=drive_link)

# + [markdown] id="eC_GF3vnTBR8"
# ### A continuación entreno un modelo **CBOW** con la librería Word2Vec. Esto porque tuve problemas para recuperar los modelos vistos en clase. Este entrenamiento lo hice con las mismas oraciones que se usaron para calcular la matriz de co-ocurrencias.

# + id="_DK36gMMe0YH"
# Train the CBOW model
N = 300
model_cbow = Word2Vec(sentences, vector_size=N, window=5, min_count=1, sg=0) # sg=0 for CBOW

# Create a list to store word vectors
word_vectors = []

# Iterate through the vocabulary and append the vectors to the list
for word in model_cbow.wv.index_to_key:
    word_vectors.append([word] + model_cbow.wv[word].tolist())

# Create a pandas DataFrame from the list of word vectors
cbow = pd.DataFrame(word_vectors, columns=["word"] + [f"dim_{i}" for i in range(N)])
cbow = cbow.set_index('word')

# + colab={"base_uri": "https://localhost:8080/", "height": 475} id="cwcDUW9qOuWE" outputId="3db88e94-0ce9-4d6b-9c32-2ab2c1785d19"
CBOW

# + [markdown] id="w8BuZlDoTjoe"
# ## ¿Cómo **comparar** los **embeddings** de ambos modelos? Usando las primeras dos componentes de un análisis de **PCA** sobre las matrices de embeddings.

# + id="pu8pLR0iS_fU"
from sklearn.decomposition import PCA

# + id="a80t-xW9quqk"
from sklearn.preprocessing import StandardScaler


# + id="MHEFHx99G40J"
def pca_graph(Comatrix1,Comatrix2,N,random_indices):

  '''
  Esta función genera un gráfico comparativo de la representación
  de embeddings de palabras en el Vocabulario mediante el uso
  de las dos primeras componentes de PCA.

  Parámetros:
  Comatrix 1 y 2: las que se van a coomparar sus embeddings
  N: El número de palabras que se van a comparar
  radom_indices: Son los indices aleatorios para seleccionar los embeddings

  Resultado:
  Un gráfico comparativo: los puntos en azul son de la comatrix1 y los rojos
  de la comatrix2
  '''

  pca = PCA(n_components=2)
  scaler = StandardScaler()

  transformed_matrix1 = pca.fit_transform(Comatrix1)
  random_elements1 = transformed_matrix1[random_indices, :]
  transformed_matrix2 = pca.fit_transform(Comatrix2)
  random_elements2 = transformed_matrix2[random_indices, :]

  normalized_elements1 = scaler.fit_transform(random_elements1)
  normalized_elements2 = scaler.fit_transform(random_elements2)

  #vocabulario_aux = [Vocabulario[i] for i in random_indices]

  plt.scatter(normalized_elements1[:, 0], normalized_elements1[:, 1],color='blue')
  #for i, word in enumerate(vocabulario):
   # plt.annotate(word, xy=(normalized_elements1[i, 0], normalized_elements1[i, 1]))

  plt.scatter(normalized_elements2[:, 0], normalized_elements2[:, 1]+6,color='red')
  #for i, word in enumerate(vocabulario):
    #plt.annotate(word, xy=(normalized_elements2[i, 0], normalized_elements2[i, 1]+6))

  plt.show()



# + id="DhIIVNmuGoAB"
N=100
random_indices = np.random.choice(len(Vocabulario), size=N, replace=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 430} id="nSvK7cchB-lZ" outputId="51c08ef6-65a9-4806-e699-248f6502ac00"
pca_graph(mut_inf,cbow,N,random_indices)

# + [markdown] id="7Xh0U3y8T689"
# ## **Reflexiones**
#
# ### Lo más importante que noté después de realizar varios experimentos fue que en general las representaciones de los vectores de CBOW tenían mayor variabilidad mientras que los de la matriz de co-ocurrencias no. Esto se traduce en que se podría calcular mejor ciertas métricas importantes como la similitud mejor en el modelo Cbow que en la matriz de co-ocurrencias.
#
# ### Sin embargo, puede que el hecho de que  las componentes estén en un espacio más reducido pueda estar ocultando información importante.
#
# ### Otro aspecto es que me da la impresión de que el Cbow tiene captura un poco mejor la riqueza o semajanzas y diferencias que existen en las palabras pues pareciera que genera más cumulos pequeños que la matriz de co-ocurrencias.

# + id="v7pxcLSL-cRP"

