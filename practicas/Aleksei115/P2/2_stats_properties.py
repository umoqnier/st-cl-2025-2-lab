# %% [markdown]
# # 2. Propiedades estadísticas del lenguaje

# %% [markdown]
# ## Objetivos

# %% [markdown]
# - Mostrar el uso de CFG y derivados
#     - Ejemplos de parseo de dependencias
# - Ejemplificar etiquetado NER usando bibliotecas existentes
# - Explorar propiedades estadísticas del lenguaje natural y observar los siguientes fenomenos:
#     - La distribución de Zipf
#     - La distribución de Heap
# 
# - Implementar bolsas de palabras
#     - Aplicar *TF.IDF*

# %% [markdown]
# ## Perspectivas formales

# %% [markdown]
# - Fueron el primer acercamiento al procesamiento del lenguaje natural. Sin embargo tienen varias **desventajas**
# - Requieren **conocimiento previo de la lengua**
# - Las herramientas son especificas de la lengua
# - Los fenomenos que se presentan son muy amplios y difícilmente se pueden abarcar con reglas formales (muchos casos especiales)
# - Las reglas tienden a ser rigidas y no admiten incertidumbre en el resultado

# %% [markdown]
# ### Sintaxis

# %% [markdown]
# ![](https://imgs.xkcd.com/comics/formal_languages_2x.png)
# 
# **[audience looks around] 'What just happened?' 'There must be some context we're missing.'**

# %% [markdown]
# #### Parsing basado en reglas

# %% [markdown]
# - Gramaticas libres de contexto:
# 
# $G = (T, N, O, R)$
# * $T$ símbolos terminales.
# * $N$ símbolos no terminales.
# * $O$ simbolo inicial o nodo raíz.
# * $R$ reglas de la forma $X \longrightarrow \gamma$ donde $X$ es no terminal y $\gamma$ es una secuencia de terminales y no terminales

# %%
from IPython.display import display, HTML #Resuelve el error de ImportError al ejecutar displacy con Jupyter Notebook
import nltk

# %%
plain_grammar = """
S -> NP VP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
PP -> P NP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
"""

# %%
grammar = nltk.CFG.fromstring(plain_grammar)
# Cambiar analizador y trace
analyzer = nltk.ChartParser(grammar)

sentence = "I shot an elephant in my pajamas".split()
trees = analyzer.parse(sentence)

# %%
for tree in trees:
    print(tree, type(tree))
    print('\nBosquejo del árbol:\n')
    print(tree.pretty_print(unicodelines=True, nodedist=1)) 

# %% [markdown]
# ## Perspectiva estadística

# %% [markdown]
# - Puede integrar aspectos de la perspectiva formal
# - Lidia mejor con la incertidumbre y es menos rigida que la perspectiva formal
# - No requiere conocimiento profundo de la lengua. Se pueden obtener soluciones de forma no supervisada

# %% [markdown]
# ## Modelos estadísticos

# %% [markdown]
# - Las **frecuencias** juegan un papel fundamental para hacer una descripción acertada del lenguaje
# - Las frecuencias nos dan información de la **distribución de tokens**, de la cual podemos estimar probabilidades.
# - Existen **leyes empíricas del lenguaje** que nos indican como se comportan las lenguas a niveles estadísticos
# - A partir de estas leyes y otras reglas estadísticas podemos crear **modelos del lenguaje**; es decir, asignar probabilidades a las unidades lingüísticas

# %% [markdown]
# ### Probabilistic Context Free Grammar

# %% [markdown]
# ###### Apunte
# 
# Podemos asignar probabilidades a cada regla de producción del árbol sintáctico.
# 
# Esto es referente al algoritmo viterbi el cuál encuentra el árbol sintáctico más probable para cada oración, ya que por cada oración podemos tener una infinidad de posibles árboles, este algoritmo propone la solución mediante el enfoque estadístico

# %%
taco_grammar = nltk.PCFG.fromstring("""
O    -> FN FV     [0.7]
O    -> FV FN     [0.3]
FN   -> Sust      [0.6]
FN   -> Det Sust  [0.4]
FV   -> V FN      [0.8]
FV   -> FN V      [0.2]
Sust -> 'Juan'    [0.5]
Sust -> 'tacos'   [0.5]
Det  -> 'unos'    [1.0]
V    -> 'come'    [1.0]
""")
viterbi_parser = nltk.ViterbiParser(taco_grammar)

# %%
sentences = [
    "Juan come unos tacos",
    "unos tacos Juan come"
]
for sent in sentences:
    for tree in viterbi_parser.parse(sent.split()):
        print(tree)
        print("Versión bosque")
        tree.pretty_print(unicodelines=True, nodedist=1)

# %% [markdown]
# ### Parseo de dependencias

# %% [markdown]
# Un parseo de dependencias devuelve las dependencias que se dan entre los tokens de una oración. Estas dependencias suelen darse entre pares de tokens. Esto es, que relaciones tienen las palabras con otras palabras.

# %% [markdown]
# ##### Freeling - https://nlp.lsi.upc.edu/freeling/demo/demo.php

# %%
import spacy
from spacy import displacy

# %%
!python -m spacy download es_core_news_md

# %%
nlp = spacy.load("es_core_news_md")

# %%
doc = nlp("La niña come un suani")

# %%
html = displacy.render(doc, style ='dep', jupyter = False)
display(HTML(html))

# %%
for chunk in doc.noun_chunks:
    print("text::", chunk.text)
    print("root::", chunk.root.text)
    print("root dep::", chunk.root.dep_)
    print("root head::", chunk.root.head.text)
    print("="*10)

# %%
!python -m spacy download es_core_news_md

# %%
for token in doc:
    print("token::", token.text)
    print("dep::", token.dep_)
    print("head::", token.head.text)
    print("head POS::", token.head.pos_)
    print("CHILDS")
    print([child for child in token.children])
    print("="*10)

# %% [markdown]
# #### Named Entity Recognition (NER)

# %% [markdown]
# El etiquetado NER consiste en identificar "objetos de la vida real" como organizaciones, paises, personas, entre otras y asignarles su etiqueta correspondiente. Esta tarea es del tipo *sequence labeling* ya que dado un texto de entrada el modelo debe identificar los intervalos del texto y etiquetarlos adecuadamente con la entidad que le corresponde. Veremos un ejemplo a continuación.

# %%
!pip install datasets

# %%
from datasets import load_dataset

# %%
data = load_dataset("alexfabbri/multi_news", trust_remote_code=True)

# %%
# Explorar data
data?

# %%
!python -m spacy download en_core_web_md

# %%
nlp = spacy.load("en_core_web_md")

# %% [markdown]
# Tomamos 3 textos de los datos, y por cada doc hace uso se aplica NER, nos muestra el texto de la entidad, donde inicia y donde termina y la etiqueta que se le asigno

# %%
import random

random.seed(42)
corpus = random.choices(data["train"]["summary"], k=3)
docs = list(nlp.pipe(corpus))
for j, doc in enumerate(docs):
    print(f"DOC #{j+1}")
    doc.user_data["title"] = " ".join(doc.text.split()[:10])
    for i, ent in enumerate(doc.ents):
        print(" -"*10, f"Entity #{i}")
        print(f"\tTexto={ent.text}")
        print(f"\tstart/end={ent.start_char}-{ent.end_char}")
        print(f"\tLabel={ent.label_}")


# %%
html = displacy.render(docs, style="ent", jupyter = False)
display(HTML(html))

# %% [markdown]
# Esta celda nos explica a que se refiere a cada uno de las etiquetas que NER le da a las entidades, en este caso ORG son 'Companies, agencies, institutions, etc.'

# %%
spacy.explain("ORG")

# %% [markdown]
# [Available labels](https://spacy.io/models/en)

# %% [markdown]
# ## Leyes estadísticas

# %%
# Bibliotecas
from collections import Counter
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [10, 6]
import numpy as np
import pandas as pd

# %%
mini_corpus = """Humanismo es un concepto polisémico que se aplica tanto al estudio de las letras humanas, los
estudios clásicos y la filología grecorromana como a una genérica doctrina o actitud vital que
concibe de forma integrada los valores humanos. Por otro lado, también se denomina humanis-
mo al «sistema de creencias centrado en el principio de que las necesidades de la sensibilidad
y de la inteligencia humana pueden satisfacerse sin tener que aceptar la existencia de Dios
y la predicación de las religiones», lo que se aproxima al laicismo o a posturas secularistas.
Se aplica como denominación a distintas corrientes filosóficas, aunque de forma particular,
al humanismo renacentista1 (la corriente cultural europea desarrollada de forma paralela al
Renacimiento a partir de sus orígenes en la Italia del siglo XV), caracterizado a la vez por su
vocación filológica clásica y por su antropocentrismo frente al teocentrismo medieval
"""
words = mini_corpus.replace("\n", " ").split(" ")
len(words)

# %%
vocabulary = Counter(words)
vocabulary.most_common(10)

# %%
len(vocabulary)

# %%
def get_frequencies(vocabulary: Counter, n: int) -> list:
    return [_[1] for _ in vocabulary.most_common(n)]

def plot_frequencies(frequencies: list, title="Freq of words", log_scale=False):
    x = list(range(1, len(frequencies)+1))
    plt.plot(x, frequencies, "-v")
    plt.xlabel("Freq rank (r)")
    plt.ylabel("Freq (f)")
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(title)

# %%
frequencies = get_frequencies(vocabulary, 100)
plot_frequencies(frequencies)

# %%
plot_frequencies(frequencies, log_scale=True)

# %% [markdown]
# **¿Qué pasará con más datos? 📊**

# %% [markdown]
# ### Ley Zipf

# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %% [markdown]
# Exploraremos el Corpus de Referencia del Español Actual [CREA](https://www.rae.es/banco-de-datos/crea/crea-anotado)

# %%
corpus_freqs = pd.read_csv("./crea_frecs.txt", sep=" ")

# %%
corpus_freqs.head(15)

# %%
corpus_freqs.iloc[0]

# %%
corpus_freqs[corpus_freqs["word"] == "barriga"]

# %%
corpus_freqs["freq"].plot(marker="o")
plt.title('Ley de Zipf en el CREA')
plt.xlabel('rank')
plt.ylabel('freq')
plt.show()

# %%
corpus_freqs['freq'].plot(loglog=True, legend=False)
plt.title('Ley de Zipf en el CREA (log-log)')
plt.xlabel('log rank')
plt.ylabel('log frecuencia')
plt.show()

# %% [markdown]
# - Notamos que las frecuencias entre lenguas siguen un patrón
# - Pocas palabras (tipos) son muy frecuentes, mientras que la mayoría de palabras ocurren pocas veces
# 
# De hecho, la frecuencia de la palabra que ocupa la posición r en el rank, es proporcional a $\frac{1}{r}$ (La palabra más frecuente ocurrirá aproximadamente el doble de veces que la segunda palabra más frecuente en el corpus y tres veces más que la tercer palabra más frecuente del corpus, etc)
# 
# $$f(w_r) \propto \frac{1}{r^α}$$
# 
# Donde:
# - $r$ es el rank que ocupa la palabra en el corpus
# - $f(w_r)$ es la frecuencia de la palabra en el corpus
# - $\alpha$ es un parámetro, el valor dependerá del corpus o fenómeno que estemos observando

# %% [markdown]
# #### Formulación de la Ley de Zipf:

# %% [markdown]
# $f(w_{r})=\frac{c}{r^{\alpha }}$
# 
# En la escala logarítimica:
# 
# $log(f(w_{r}))=log(\frac{c}{r^{\alpha }})$
# 
# $log(f(w_{r}))=log (c)-\alpha log (r)$

# %% [markdown]
# #### ❓ ¿Cómo estimar el parámetro $\alpha$?

# %% [markdown]
# Podemos hacer una regresión lineal minimizando la suma de los errores cuadráticos:
# 
# $J_{MSE}=\sum_{r}^{}(log(f(w_{r}))-(log(c)-\alpha log(r)))^{2}$

# %%
from scipy.optimize import minimize

ranks = np.array(corpus_freqs.index) + 1
frecs = np.array(corpus_freqs['freq'])

# Inicialización
a0 = 1

# Función de minimización:
func = lambda a: sum((np.log(frecs)-(np.log(frecs[0])-a*np.log(ranks)))**2)

# Apliando minimos cuadrados
a_hat = minimize(func, a0).x[0]

print('alpha:', a_hat, '\nMSE:', func(a_hat))

# %%
def plot_generate_zipf(alpha: np.float64, ranks: np.array, freqs: np.array) -> None:
    plt.plot(np.log(ranks),  np.log(freqs[0]) - alpha*np.log(ranks), color='r', label='Aproximación Zipf')

# %%
plot_generate_zipf(a_hat, ranks, frecs)
plt.plot(np.log(ranks),np.log(frecs), color='b', label='Distribución CREA')
plt.xlabel('log ranks')
plt.ylabel('log frecs')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# %% [markdown]
# ### Ley de Heap

# %% [markdown]
# Relación entre el número de **tokens** y **tipos** de un corpus
# 
# $$T \propto N^b$$
# 
# Dónde:
# 
# - $T = $ número de tipos
# - $N = $ número de tokens
# - $b = $ parámetro  

# %% [markdown]
# - **TOKENS**: Número total de palabras dentro del texto (incluidas repeticiones)
# - **TIPOS**: Número total de palabras únicas en el texto

# %% [markdown]
# #### 📊 Ejercicio: Muestra el plot de tokens vs types para el corpus CREA
# 
# **HINT:** Obtener tipos y tokens acumulados

# %%
# PLOT tokens vs types
total_tokens = corpus_freqs["freq"].sum()
total_types = len(corpus_freqs)

# %%
corpus_sorted = corpus_freqs.sort_values(by="freq", ascending=False)
corpus_sorted["cum_tokens"] = corpus_sorted["freq"].cumsum()
corpus_sorted["cum_types"] = range(1, total_types +1)

# %%
# Plot de la ley de Heap
plt.plot(corpus_sorted['cum_types'], corpus_sorted['cum_tokens'])
plt.xscale("log")
plt.yscale("log")
plt.xlabel('Types')
plt.ylabel('Tokens')
plt.title('Ley de Heap')
plt.show()

# %% [markdown]
# ### Representaciones vectoriales estáticas (estáticos)

# %% [markdown]
# - Buscamos una forma de mapear textos al **espacio vectorial**. Tener una representación numerica permite su procesamiento.
#     - Similitud de docs
#     - Clasificacion (agrupamiento)
# - Veremos el enfoque de la Bolsa de Palabras (Bag of Words)
#    - Matriz de documentos-terminos
#    - Cada fila es un vector con $N$ features donde las features serán el vocabulario del corpus

# %% [markdown]
# <center>
# <img src="https://preview.redd.it/sqkqsuit7o831.jpg?width=1024&auto=webp&s=2d18d38fe9d04a4a62c9a889e7b34ef14b425630" width=500></center>

# %%
import gensim

# %%
doc_1 = "Augusta Ada King, condesa de Lovelace (Londres, 10 de diciembre de 1815-íd., 27 de noviembre de 1852), registrada al nacer como Augusta Ada Byron y conocida habitualmente como Ada Lovelace, fue una matemática y escritora británica, célebre sobre todo por su trabajo acerca de la computadora mecánica de uso general de Charles Babbage, la denominada máquina analítica. Fue la primera en reconocer que la máquina tenía aplicaciones más allá del cálculo puro y en haber publicado lo que se reconoce hoy como el primer algoritmo destinado a ser procesado por una máquina, por lo que se le considera como la primera programadora de ordenadores."
doc_2 = "Brassica oleracea var. italica, el brócoli,1​ brécol2​ o bróquil3​ del italiano broccoli (brote), es una planta de la familia de las brasicáceas. Existen otras variedades de la misma especie, tales como: repollo (B. o. capitata), la coliflor (B. o. botrytis), el colinabo (B. o. gongylodes) y la col de Bruselas (B. o. gemmifera). El llamado brócoli chino o kai-lan (B. o. alboglabra) es también una variedad de Brassica oleracea."
doc_3 = "La bicicleta de piñón fijo, fixie o fixed es una bicicleta monomarcha, que no tiene piñón libre, lo que significa que no tiene punto muerto; es decir, los pedales están siempre en movimiento cuando la bicicleta está en marcha. Esto significa que no se puede dejar de pedalear, ya que, mientras la rueda trasera gire, la cadena y los pedales girarán siempre solidariamente. Por este motivo, se puede frenar haciendo una fuerza inversa al sentido de la marcha, y también ir marcha atrás."

# %%
documents = [doc_1, doc_2, doc_3]

# %%
from gensim.utils import simple_preprocess

def sent_to_words(sentences: list[str]) -> list[list[str]]:
    """Function convert sentences to words

    Use the tokenizer provided by gensim using
    `simple_process()` which remove punctuation and converte
    to lowercase (`deacc=True`)
    """
    return [simple_preprocess(sent, deacc=True) for sent in sentences]


# %%
docs_tokenized = sent_to_words(documents)
docs_tokenized[0][:10]

# %%
from gensim.corpora import Dictionary

gensim_dic = Dictionary()
bag_of_words_corpus = [gensim_dic.doc2bow(doc, allow_update=True) for doc in docs_tokenized]

# %%
type(gensim_dic)

# %%
for k, v in gensim_dic.iteritems():
    print(k, v)

# %%
print(len(bag_of_words_corpus))
bag_of_words_corpus[0]

# %%
def bag_to_dict(bag_of_words: list, gensim_dic: Dictionary, titles: list[str]) -> list:
    data = {}
    for doc, title in zip(bag_of_words, titles):
        data[title] = dict([(gensim_dic[id], freq) for id, freq in doc])
    return data

# %%
data = bag_to_dict(bag_of_words_corpus, gensim_dic, titles=["ADA", "BROCOLI", "FIXED"])

# %%
data

# %%
import pandas as pd

doc_matrix_simple = pd.DataFrame(data).fillna(0).astype(int).T

# %%
doc_matrix_simple

# %% [markdown]
# - Tenemos una matrix de terminos-frecuencias ($tf$). Es decir cuantas veces un termino aparece en cierto documento.
# - Una variante de esta es una **BoW** binaria. ¿Cómo se vería?
# 

# %% [markdown]
# **¿Ven algun problema?**

# %% [markdown]
# - Palabras muy frecuentes que no aportan signifiancia
# - Los pesos de las palabras son tratados de forma equitativa
#     - Palabras muy frecuentes opacan las menos frecuentes y con mayor significado (semántico) en nuestros documentos
# - Las palabras frecuentes no nos ayudarian a discriminar por ejemplo entre documentos

# %% [markdown]
# #### *Term frequency-Inverse Document Frequency* (TF-IDF) al rescate
# 
# <center><img src="https://media.tenor.com/Hqyg8s_gh5QAAAAd/perfectly-balanced-thanos.gif" height=250></center>

# %% [markdown]
# - Metodo de ponderación creado para algoritmos de Information Retrieval
# - Bueno para clasificación de documentos y clustering
# - Se calcula con la multiplicacion $tf_{d,t} \cdot idf_t$
# 
# Donde:
#   - $tf_{d,t}$ es la frecuencia del termino en un documento $d$
#   - $idf_t$ es la frecuencia inversa del termino en toda la colección de documentos. Se calcula de la siguiente forma:
# 
# $$idf_t = log_2\frac{N}{df_t}$$
# 
# Entonces:
# 
# $$tf\_idf(d,t) = tf_{d,t} ⋅ \log_2\frac{N}{df_t}$$

# %% [markdown]
# #### 🧮 Ejercicio: Aplica TF-IDF usando gensim
# 
# **HINT:** https://radimrehurek.com/gensim/models/tfidfmodel.html

# %%
from gensim.models import TfidfModel

tfidf = TfidfModel(bag_of_words_corpus, smartirs="ntc")

# %%
tfidf[bag_of_words_corpus[0]]

# %%
def bag_to_dict_tfidf(bag_of_words: list, gensim_dic: Dictionary, titles: list[str]) -> list:
    data = {}
    tfidf = TfidfModel(bag_of_words, smartirs="ntc")
    for doc, title in zip(tfidf[bag_of_words], titles):
        data[title] = dict([(gensim_dic[id], freq) for id, freq in doc])
    return data

# %%
data = bag_to_dict_tfidf(bag_of_words_corpus, gensim_dic, titles=["ADA", "BROCOLI", "FIXED"])

# %%
data

# %%
doc_matrix_tfidf = pd.DataFrame(data).fillna(0).T

# %%
doc_matrix_tfidf

# %% [markdown]
# #### Calculando similitud entre vectores

# %% [markdown]
# <center><img src="https://cdn.acidcow.com/pics/20130320/people_who_came_face_to_face_with_their_doppelganger_19.jpg" width=500></center>

# %% [markdown]
# La forma estandar de obtener la similitud entre vectores para **BoW** es con la distancia coseno entre ellos
# 
# $$cos(\overrightarrow{v},\overrightarrow{w}) = \frac{\overrightarrow{v} \cdot\overrightarrow{w}}{|\overrightarrow{v}||\overrightarrow{w}|}$$
# 
# Aunque hay muchas más formas de [calcular la distancia](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) entre vectores

# %%
from sklearn.metrics.pairwise import cosine_similarity

doc_1 = doc_matrix_tfidf.loc["BROCOLI"].values.reshape(1, -1)
doc_2 = doc_matrix_tfidf.loc["FIXED"].values.reshape(1, -1)
cosine_similarity(doc_1, doc_2)

# %% [markdown]
# #### Agregando más documentos a nuestra bolsa

# %% [markdown]
# ![](https://media.tenor.com/55hA4TgUrOMAAAAM/bag-bags.gif)

# %%
def update_bow(doc: str, bag_of_words: list, gensim_dic: Dictionary) -> pd.DataFrame:
    words = simple_preprocess(doc, deacc=True)
    bag_of_words.append(gensim_dic.doc2bow(words, allow_update=True))
    return bag_of_words

# %%
#sample_doc = "Las bicicletas fixie, también denominadas bicicletas de piñón fijo, son bicis de una sola marcha, de piñón fijo, y sin punto muerto, por lo que se debe avanzar, frenar y dar marcha atrás con el uso de los pedales. La rueda de atrás gira cuando giran los pedales. Si pedaleas hacia delante, avanzas; si paras los pedales, frenas y si pedaleas hacia atrás, irás marcha atrás. Esto requiere de un entrenamiento añadido que la bicicleta con piñón libre no lo necesita. No obstante, las bicicletas fixie tienen muchísimas ventajas."
sample_doc = "El brócoli o brécol es una planta de la familia de las brasicáceas, como otras hortalizas que conocemos como coles. Está por tanto emparentado con verduras como la coliflor, el repollo y las diferentes coles lisas o rizadas, incluyendo el kale o las coles de Bruselas."

# %%
new_bag = update_bow(sample_doc, bag_of_words_corpus.copy(), gensim_dic)
len(new_bag)

# %%
for k, v in gensim_dic.iteritems():
    print(k, v)

# %%
new_data = bag_to_dict_tfidf(new_bag, gensim_dic, ["ADA", "BROCOLI", "FIXED", "SAMPLE"])

# %%
new_doc_matrix_tfidf = pd.DataFrame(new_data).fillna(0).T
new_doc_matrix_tfidf

# %% [markdown]
# #### 👯‍♂️ Ejercicio: Calcula la similitud del nuevo documento con el resto de documentos

# %%
doc_sample_values = new_doc_matrix_tfidf.loc["SAMPLE"].values.reshape(1, -1)

doc_titles = ["ADA", "BROCOLI", "FIXED"]
for i, doc_title in enumerate(doc_titles):
    current_doc_values = new_doc_matrix_tfidf.loc[doc_title].values.reshape(1, -1)
    print(f"Similarity beetwen SAMPLE/{doc_title}= {cosine_similarity(current_doc_values, doc_sample_values)}")

# %% [markdown]
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
# Ejercicio 1

# %% [markdown]
# Para resolver este ejercicio, la ley de Zipf nos dice que la frequencia de la palabra con rango r será igual a un 1/r**a donde r es el rango de la palabra, es decir el lugar que ocupa de la palabra más frequente a la menos frequente, esto debe demostrar que:
# 
# La palabra más frecuente aparece aproximadamente el doble de veces que la segunda palabra más frecuente, la segunda palabra más frequente aparece el doble de veces que la cuarta más frequente

# %%
#Generamos nuestro lenguaje artificial, tomare como inicio el corpus de los ejemplos
import random
from collections import defaultdict
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def generate_language():
    language = []
    for i in range(10000):
        language.append(generate_word(random.randint(1,7)))
    return language

def generate_word(len_word):
    word = ''
    for i in range(len_word):
        word += alphabet[random.randint(0, len(alphabet)-1)]
    return word


new_language = generate_language()
dict_frequencies = defaultdict(int)

for word in new_language:
    dict_frequencies[word] += 1

#Ordenamos por frequencia
dict_frequencies = sorted(dict_frequencies.items(), key=lambda x: x[1], reverse=True)

#Creamos un diccionario con rango y frequencia
dict_frequencies_rank = [(i+1, dict_frequencies[i][1]) for i in range(len(dict_frequencies))]

#Imprimimos las 10 palabras mas frecuentes
for i in range(10):
    print(dict_frequencies[i])

#Mostramos el gráfico de las frequencias log-log
import matplotlib.pyplot as plt
plt.loglog([i[0] for i in dict_frequencies_rank], [i[1] for i in dict_frequencies_rank], color='b', label='frecuencia')

plt.title('Ley de Zipf en el Corpus creado')
plt.xlabel('rank')
plt.ylabel('freq')
plt.show()



# %% [markdown]
# Se puede observar un comportamiento similar a los ejemplos dentro de la práctica, sin embargo el utilizar un lenguaje creado aleatoriamente hace que este comportamiento no siga una linea tan suave como en el ejemplo de CREA, podríamos implementar lagunas soluciones para la creación del lenguaje basado en reglas, pero al final obtendríamos el mismo resultado

# %% [markdown]
# ##### Ejercicio 2

# %% [markdown]
# Para este ejercicio se elijen 3 datasets:
# 
# - Tweets en español
# - Dataset de speech
# - Descripcion de peliculas

# %%
from datasets import load_dataset
import spacy
from spacy import displacy
import random
from IPython.display import display, HTML

random.seed(42)


ds1 = load_dataset("pysentimiento/spanish-tweets")
ds2 = load_dataset("PereLluis13/spanish_speech_text")
ds3 = load_dataset("mathigatti/spanish_imdb_synopsis")

# %%
nlp = spacy.load("es_core_news_md")

corpus = random.choices(ds1['test']['text'], k=6)
docs = list(nlp.pipe(corpus))

html = displacy.render(docs, style="ent", jupyter = False)
display(HTML(html))

# %% [markdown]
# Para este primer dataset se eligió usar el conjunto de test ya que el costo computacional dentro del conjunto de entrenamiento, incluso para la selección de pocas entidades era demasiado, de los resultados podemos observar que el NER no tiene una muy buena identificación de entidades, ya que para los usuarios los identifica como "Locaciones", vemos un buen etiquetado para la organización "Emirates" la cual se refiere a una aerolinea, mi hipotésis es que el modelo que estamos utilizando esta muy enfocado a noticias, lo cual, hace que el NER sea bueno para identificar organizaciones o personas, pero vemos que etiqueta la cadena "jajaja" como un miscelaneos lo cual no es muy útil, un modelo entrenado para tweets podría aumentar el grado de eficiencia

# %%
spacy.explain("PER")

# %%
nlp = spacy.load("es_core_news_md")

corpus = random.choices(ds2['train']['sentence'], k=7)
docs = list(nlp.pipe(corpus))

html = displacy.render(docs, style="ent", jupyter = False)
display(HTML(html))

# %% [markdown]
# En el dataset sobre speech se observa un mejor desempeño ya que el corpus esta enfocado a un lenguaje que es utilizado en las noticias, vemos que hay un buen reconocimiento de locaciones, de organizaciones, personas y hay una etiquena miscelaneos, lo cual creó que no debería ir ahi, sin embargo, el desempeño mejora notablemente

# %%
nlp = spacy.load("es_core_news_md")

corpus = random.choices(ds3['train']['description'], k=10)
docs = list(nlp.pipe(corpus))

html = displacy.render(docs, style="ent", jupyter = False)
display(HTML(html))

# %% [markdown]
# Para el último dataset vemos una mejoria aún más grande, ya que las descripciones de peliculas suelen tener un mayor número de locaciones y personas, además, nuestro modelo perfectamente puede etiquetar sin tantas equivocaciones, para mí observando la frase "Un agente del FBI ORG sube a un avión lleno de serpientes venenosas, liberadas intencionalmente para matar a un testigo que está volando de Honolulu LOC a Los Ángeles LOC para testificar contra un jefe de la mafia." podría el modelo etiqueta "jefe de la mafia" como una persona, sin embargo creó que esta idea cae en la idea de la pragmática, ya que uno como persona sabe que se refiere a una persona, se observa que el modelo identifica bien a personas dados nombres propios


