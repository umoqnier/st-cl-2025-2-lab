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
#     name: python3
# ---

# + [markdown] id="sRSsHYYwVWsU"
# # Práctica 4: Modelos del Lenguaje Neuronales
#
# **Fecha de entrega: 6 de abril de 2025 11:59pm**

# + [markdown] id="L8961cBVVWsc"
# ## Ejercicio 1.

# + [markdown] id="7d0YQ51oVWsd"
# A partir del modelo entrenado:
#
# - Sacar los embeddings de las palabras del vocabulario
#
# - Visualizar en 2D los embeddings de algunas palabras (quizá las más frecuentes, excluyendo stopwords)
#
# - Seleccione algunas palabras y verifique sí realmente codifican nociones semánticas, e,g, similitud semántica con similitud coseno entre dos vectores, analogías por medios de operaciones de vectores

# + id="_DG7ZU1FVWse"
# Bibliotecas
import nltk

# + [markdown] id="4xlFZEheVWsf"
# ### Definición del modelo

# + colab={"base_uri": "https://localhost:8080/"} id="nMEl_-1vVWsg" outputId="b56d1a3b-d94a-4982-c70c-87d59851e1fa"
nltk.download('reuters')
nltk.download('punkt_tab')

from nltk.corpus import reuters
from nltk import ngrams


# + id="rblv1TUBVWsh"
def preprocess_corpus(corpus: list[str]) -> list[str]:
    """Función de preprocesamiento

    Esta función está diseñada para preprocesar
    corpus para modelos del lenguaje neuronales.
    Agrega tokens de inicio y fin, normaliza
    palabras a minusculas
    """
    preprocessed_corpus = []
    for sent in corpus:
        result = [word.lower() for word in sent]
        # Al final de la oración
        result.append("<EOS>")
        result.insert(0, "<BOS>")
        preprocessed_corpus.append(result)
    return preprocessed_corpus


# + id="3hyGoY4DVWsi"
def get_words_freqs(corpus: list[list[str]]):
    """Calcula la frecuencia de las palabras en un corpus"""
    words_freqs = {}
    for sentence in corpus:
        for word in sentence:
            words_freqs[word] = words_freqs.get(word, 0) + 1
    return words_freqs


# + id="xOwIP6PjVWsi"
UNK_LABEL = "<UNK>"
def get_words_indexes(words_freqs: dict) -> dict:
    """Calcula los indices de las palabras dadas sus frecuencias"""
    result = {}
    for idx, word in enumerate(words_freqs.keys()):
        # Happax legomena happends
        if words_freqs[word] == 1:
            # Temp index for unknowns
            result[UNK_LABEL] = len(words_freqs)
        else:
            result[word] = idx

    return {word: idx for idx, word in enumerate(result.keys())}, {idx: word for idx, word in enumerate(result.keys())}


# + id="l4a71aixVWsj"
corpus = preprocess_corpus(reuters.sents())

# + id="WtHKR7xOVWsk"
words_freqs = get_words_freqs(corpus)

# + colab={"base_uri": "https://localhost:8080/"} id="2aZcbBzuVWsl" outputId="55afdf78-021b-45aa-ca57-04b6ea7a6313"
count = 0
for word, freq in words_freqs.items():
    if freq == 1 and count <= 10:
        print(word, freq)
        count += 1

# + id="cO3SQCRwVWsl"
words_indexes, index_to_word = get_words_indexes(words_freqs)

# + colab={"base_uri": "https://localhost:8080/"} id="a5xANWO9VWsl" outputId="0961e296-ce5f-44e9-b39a-0b475db1ac78"
words_indexes["the"]

# + colab={"base_uri": "https://localhost:8080/"} id="isGj9fJLh6Nj" outputId="657eb1e4-873f-4c33-ac53-9bc3198b2703"
words_indexes["women"]

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="tC3ece-1VWsl" outputId="26e3930e-53bf-4f63-f62a-02d5995a45f2"
index_to_word[16]

# + colab={"base_uri": "https://localhost:8080/"} id="36hz2Un6VWsl" outputId="108f2613-3925-4575-d1ff-6527ab6b5707"
len(words_indexes)

# + colab={"base_uri": "https://localhost:8080/"} id="NRyBtkxKVWsm" outputId="0f95855a-1759-4b30-b711-f90d296efb5d"
len(index_to_word)


# + id="VVHNpZfjVWsm"
def get_word_id(words_indexes: dict, word: str) -> int:
    """Obtiene el id de una palabra dada

    Si no se encuentra la palabra se regresa el id
    del token UNK
    """
    unk_word_id = words_indexes[UNK_LABEL]
    return words_indexes.get(word, unk_word_id)


# + [markdown] id="JzMd3kXMVWsm"
# ### Obtenemos trigramas

# + [markdown] id="EdFbnlEdVWsm"
# Convertiremos los trigramas obtenidos a secuencias de idx, y preparamos el conjunto de entrenamiento $x$ y $y$
#
# - x: Contexto
# - y: Predicción de la siguiente palabra

# + id="PWcRxrHmVWsn"
def get_train_test_data(corpus: list[list[str]], words_indexes: dict, n: int) -> tuple[list, list]:
    """Obtiene el conjunto de train y test

    Requerido en el step de entrenamiento del modelo neuronal
    """
    x_train = []
    y_train = []
    for sent in corpus:
        n_grams = ngrams(sent, n)
        for w1, w2, w3 in n_grams:
            x_train.append([get_word_id(words_indexes, w1), get_word_id(words_indexes, w2)])
            y_train.append([get_word_id(words_indexes, w3)])
    return x_train, y_train


# + [markdown] id="S6J6oxtcVWsn"
# ### Preparando Pytorch
#
# $x' = e(x_1) \oplus e(x_2)$
#
# $h = \tanh(W_1 x' + b)$
#
# $y = softmax(W_2 h)$

# + id="wkPwIdbgVWsn"
# cargamos bibliotecas
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# + id="yqBjE4tTVWsn"
# Setup de parametros
EMBEDDING_DIM = 200
CONTEXT_SIZE = 2
BATCH_SIZE = 256
H = 100
torch.manual_seed(42)
# Tamaño del Vocabulario
V = len(words_indexes)

# + id="sbswGTW7VWso"
x_train, y_train = get_train_test_data(corpus, words_indexes, n=3)

# + id="RGId0f8dVWso"
import numpy as np

train_set = np.concatenate((x_train, y_train), axis=1)
# partimos los datos de entrada en batches
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE)


# + [markdown] id="EgrEeSf6VWso"
# ### Creamos la arquitectura del modelo

# + id="fwkM19dKVWso"
# Trigram Neural Network Model
class TrigramModel(nn.Module):
    """Clase padre: https://pytorch.org/docs/stable/generated/torch.nn.Module.html"""

    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size)

    def forward(self, inputs):
        # x': concatenation of x1 and x2 embeddings   -->
        # self.embeddings regresa un vector por cada uno de los índices que se les pase como entrada.
        # view() les cambia el tamaño para concatenarlos
        embeds = self.embeddings(inputs).view((-1,self.context_size * self.embedding_dim))
        # h: tanh(W_1.x' + b)  -->
        out = torch.tanh(self.linear1(embeds))
        # W_2.h                 -->
        out = self.linear2(out)
        # log_softmax(W_2.h)      -->
        # dim=1 para que opere sobre renglones, pues al usar batchs tenemos varios vectores de salida
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


# + id="egtPaaQdVWsp"
# Seleccionar la GPU si está disponible
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="83w2R4oEVWsp" outputId="96fea0d5-4bb9-47d2-82a5-f53945e78209"
device

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="CK3l99VbVWsp" outputId="8d394d27-0637-4661-d42e-a2bed8b6a576"
#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}")

# 1. Pérdida. Negative log-likelihood loss
loss_function = nn.NLLLoss()

# 2. Instanciar el modelo y enviarlo a device
model = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, H).to(device)

# 3. Optimización. ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr = 2e-3)

# ------------------------- TRAIN & SAVE MODEL ------------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    st = time.time()
    print("\n--- Training model Epoch: {} ---".format(epoch))
    for it, data_tensor in enumerate(train_loader):
        # Mover los datos a la GPU
        context_tensor = data_tensor[:,0:2].to(device)
        target_tensor = data_tensor[:,2].to(device)

        model.zero_grad()

        # FORWARD:
        log_probs = model(context_tensor)

        # compute loss function
        loss = loss_function(log_probs, target_tensor)

        # BACKWARD:
        loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print("Training Iteration {} of epoch {} complete. Loss: {}; Time taken (s): {}".format(it, epoch, loss.item(), (time.time()-st)))
            st = time.time()
'''
    # saving model
    model_path = f'/content/drive/MyDrive/8vo Semestre/LM_neuronal/model_{device}_context_{CONTEXT_SIZE}_epoch_{epoch}.dat'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved for epoch={epoch} at {model_path}")
'''


# + colab={"base_uri": "https://localhost:8080/"} id="FpYMpmryVWsp" outputId="7986ae90-ec5f-42ad-cfca-12863a111ad7"
model

# + id="2Y8EY6r-VWsq"
#model = get_model(PATH)
W1 = "<BOS>"
W2 = "my"

IDX1 = get_word_id(words_indexes, W1)
IDX2 = get_word_id(words_indexes, W2)

#Obtenemos Log probabidades p(W3|W2,W1)
probs = model(torch.tensor([[IDX1,  IDX2]]).to(device)).detach().tolist()

# + colab={"base_uri": "https://localhost:8080/"} id="XNOszK6FVWsq" outputId="1f8b1873-2546-45ba-8f2a-e9da6a8b4e34"
len(probs[0])

# + colab={"base_uri": "https://localhost:8080/"} id="V8fKOWVSVWsw" outputId="e1173c73-fca8-4ecf-c1f8-74b03e635d04"
# Creamos diccionario con {idx: logprob}
model_probs = {}
for idx, p in enumerate(probs[0]):
  model_probs[idx] = p

# Sort:
model_probs_sorted = sorted(((prob, idx) for idx, prob in model_probs.items()), reverse=True)

# Printing word  and prob (retrieving the idx):
topcandidates = 0
for prob, idx in model_probs_sorted:
  #Retrieve the word associated with that idx
  word = index_to_word[idx]
  print(idx, word, prob)

  topcandidates += 1

  if topcandidates > 10:
    break

# + colab={"base_uri": "https://localhost:8080/"} id="8ahJrYnXVWsw" outputId="fa7a6c6a-f95e-4a23-b93e-1469ca14482a"
print(index_to_word.get(model_probs_sorted[0][1]))

# + [markdown] id="rxvNPupMVWsy"
# ### Obtener embeddings

# + id="-MbF4JKubJty"
# Crear un diccionario vacío para almacenar la palabra y su embedding
word_embeddings_dict = {}

# Recorrer todos los índices en el vocabulario
for idx in range(len(index_to_word)):
    # Obtener la palabra correspondiente
    word = index_to_word[idx]
     # Mover el índice al dispositivo del modelo
    idx_tensor = torch.tensor([idx]).to(device)

    # Obtener el embedding para el índice
    embedding = model.embeddings(idx_tensor)

    # Guardar el embedding en el diccionario
    word_embeddings_dict[word] = embedding.detach().cpu().numpy().flatten()

# + colab={"base_uri": "https://localhost:8080/"} id="X93rFAGbcXdX" outputId="81e6a265-6ca1-481e-f728-4566082468b0"
# Imprimir el diccionario resultante
from itertools import islice
for word, embedding in islice(word_embeddings_dict.items(),20):
    print(f"Palabra: {word}, Embedding: {embedding[:5]}...")  # Mostrar solo los primeros 5 valores del embedding

# + [markdown] id="QM6_lFMsiZjg"
# ### Visualizar embeddings

# + id="uU-6QKzIcfAN"
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# + colab={"base_uri": "https://localhost:8080/"} id="8TRNAdfkcdb-" outputId="56ca0093-4bdf-4224-940d-f5181b1b92a5"
# Descargar las stopwords de nltk si no las tienes
nltk.download('stopwords')

# Obtener las stopwords en inglés de nltk
stopwords_en = set(stopwords.words('english'))

# Ordenar las palabras por frecuencia (de mayor a menor)
sorted_words = sorted(words_freqs.items(), key=lambda item: item[1], reverse=True)

# Seleccionar las 50 palabras más comunes, excluyendo las stopwords y con longitud > 3
top_50_words = [
                word for word, _ in sorted_words[:200]  # Seleccionar las primeras 200 para asegurarnos de que haya suficientes palabras con longitud > 3
                if word not in stopwords_en and len(word) > 3
               ][:50]  # Limitar a las primeras 50

# Filtrar los embeddings de las 50 palabras más comunes
filtered_words = {word: word_embeddings_dict[word] for word in top_50_words}

# Obtener los embeddings y las palabras
words = list(filtered_words.keys())
embeddings = np.array(list(filtered_words.values()))

# + colab={"base_uri": "https://localhost:8080/"} id="mM7yWHzLfwa6" outputId="b1818a63-ffcf-4627-d71c-53c8b5b8cd54"
len(words)

# + id="uFPml5f0csZG"
from sklearn.decomposition import PCA

# Usar PCA para reducir la dimensionalidad a 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)


# + colab={"base_uri": "https://localhost:8080/", "height": 714} id="xVCJvZ87dT1R" outputId="e4154dfb-ccb6-4685-f51c-e2f14ac7503d"
# Visualizar los resultados
plt.figure(figsize=(10, 8))

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='darkorchid', s=50)  # `s` para el tamaño de los puntos

# Añadir etiquetas por encima de los puntos
for i, word in enumerate(words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1] + 0.1),  # `+ 0.1` coloca la etiqueta por encima del punto
                 fontsize=9, ha='center', color='black')  # `ha='center'` centra la etiqueta

plt.title('Visualización 2D de los Embeddings de Palabras Más Comunes en el vocabulario\n(Sin Stopwords y Longitud > 3)\n')

plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.grid(True)
plt.show()

# + [markdown] id="YDUuQ4IWgQlY"
# ### Análisis nociones semánticas

# + id="bT6RFetSgi1W"
from sklearn.metrics.pairwise import cosine_similarity
def similitud_coseno(word1, word2):
    """Calcula la similitud coseno entre dos vectores"""

    # Obtener sus embeddings
    embedding1 = word_embeddings_dict[word1]
    embedding2 = word_embeddings_dict[word2]

    # Calcular la similitud coseno
    cosine_sim = cosine_similarity([embedding1], [embedding2])
    print(f"Similitud coseno entre {word1} y {word2}: {cosine_sim[0][0]}")


# + colab={"base_uri": "https://localhost:8080/"} id="MQu2avIagX_v" outputId="421a868f-880b-4818-89ea-66952f39c7ce"
similitud_coseno("february","january")
similitud_coseno("price","market")
similitud_coseno("march","april")
similitud_coseno("money","dollar")


# + id="Ejpp3vJ9ghX4" colab={"base_uri": "https://localhost:8080/"} outputId="70f26ca4-71fe-4670-aaf0-82dc946e65dc"
from scipy.spatial.distance import cosine

word1 = 'rich'
word2 = 'money'
word3 = 'poor'
word4 = 'capital'
word5 = 'city'
word6 = 'country'

# Obtener los embeddings
embedding_rich = word_embeddings_dict[word1]
embedding_money = word_embeddings_dict[word2]
embedding_poor = word_embeddings_dict[word3]
embedding_capital = word_embeddings_dict[word4]
embedding_city = word_embeddings_dict[word5]
embedding_country = word_embeddings_dict[word6]

# Realizar las operaciones de analogía
analogy1_result = embedding_rich - embedding_money + embedding_poor
analogy2_result = embedding_capital - embedding_city + embedding_country

# Buscar las palabras más cercanas a los resultados de las analogías
def find_most_similar(analogy_result):
    similarities = {}
    for word, embedding in word_embeddings_dict.items():
        similarity = cosine(analogy_result, embedding)
        similarities[word] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1])
    return sorted_similarities[:10]  # Retornar las 10 palabras más cercanas

# Ver los resultados de las analogías
print("Palabras más cercanas a 'rich - money + poor':")
for word, similarity in find_most_similar(analogy1_result):
    print(f"{word}: {similarity}")

print("\nPalabras más cercanas a 'capital - city + country':")
for word, similarity in find_most_similar(analogy2_result):
    print(f"{word}: {similarity}")

# + [markdown] id="wjV2YNHlQhDp"
# ### Conclusiones
#
# El modelo muestra una buena capacidad para capturar relaciones semánticas claras, como las que existen entre palabras como "february" y "january", "price" y "market", o "money" y "dollar". Sin embargo, presenta dificultades al resolver analogías más complejas, como "rich - money + poor" o "capital - city + country". Esto sugiere que el modelo aún no maneja bien las analogías semánticas. En la visualización 2D, las palabras relacionadas con el tiempo y la economía están bien agrupadas, pero algunas palabras están más alejadas, lo que indica que el modelo podría mejorar en la representación de ciertas relaciones semánticas.

# + [markdown] id="kuRuUOhPVWsy"
# ## Referencias
#
# - [Language models - Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html#generation_strategies)
# - [A Neural Probabilistic Model - Bengio](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
# - Parte del código de esta práctica fue retomado del trabajo de la Dr. Ximena Guitierrez Vasques
