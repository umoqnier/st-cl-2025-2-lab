#!/usr/bin/env python
# coding: utf-8

# # Vázquez Martínez Fredin Alberto

# In[50]:


import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from rich import print as rprint
import requests
import random 

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# In[16]:


device = "cuda" if torch.cuda.is_available() else "cpu"
rprint(f"Working on device={device}")


# ## Funciones creadas durante clase

# In[17]:


class Token(dict):
    """Modela cada renglon de un corpus en formato CoNLL
    """
    pass

t = Token(
    {
        "ID": "1",
        "FORM": "Las",
        "LEMMA": "el",
        "UPOS": "DET",
        "FEATS": "Definite=Def|Gender=Fem|Number=Plur|PronType=Art",
    }
)
rprint(t)


# In[18]:


import re

class CoNLLDictorizer:
    """Convierte un corpus en formato CoNLL a una lista de diccionarios

    Define los métodos fit, transform y fit_transform para que
    sea compatible con la api de scikit-learn.

    Parameters
    ----------
    column_names: list
        Nombre de las columnas del corpus.
        Default: ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]
    sent_sep: str
        Separador de oraciones. Default: "\n\n"
    col_sep: str
        Separador de columnas. Default: "\t+"
    """
    DEFAULT_COLS = [
        "ID",
        "FORM",
        "LEMMA",
        "UPOS",
        "XPOS",
        "FEATS",
        "HEAD",
        "DEPREL",
        "HEAD",
        "DEPS",
        "MISC",
    ]

    def __init__(self, column_names: list=DEFAULT_COLS, sent_sep="\n\n", col_sep="\t+"):
        self.column_names = column_names
        self.sent_sep = sent_sep
        self.col_sep = col_sep

    def fit(self):
        pass

    def transform(self, corpus: str) -> list[Token]:
        """Convierte un corpus en formato CoNLL a una lista de diccionarios.

        Parameters
        ----------
        corpus: str
            Corpus en formato CoNLL

        Return
        ------
        list
            Lista de diccionarios con los tokens del corpus
        """
        corpus = corpus.strip()
        sentences = re.split(self.sent_sep, corpus)
        return list(map(self._split_in_words, sentences))

    def fit_transform(self, corpus):
        return self.transform(corpus)

    def _split_in_words(self, sentence: list[str]) -> list[Token]:
        """Preprocesa una oración en formato CoNLL

        Ignora las lineas que comienzan con "#" y separa
        cada línea en un diccionario.

        Parameters
        ----------
        sentence: str
            Oracion en formato CoNLL

        Return
        ------
        list
            Lista de diccionarios con los tokens de la oración
        """
        rows = re.split("\n", sentence)
        rows = [row for row in rows if row[0] != "#"]
        return [
            Token(dict(zip(self.column_names, re.split(self.col_sep, row))))
            for row in rows
        ]


# In[68]:


def extract_features(sentence: list[str], context: int=2) -> list:
    """Extraer las features de cada oración

    Para tener siempre la misma cantidad de features
    por oración aplica un ventaneo llenando los espacios
    restantes con "<BOS>" y "<EOS>"

    Parameters
    ----------
    sentence: list[str]
        Oracion en formato CoNLL
    context: int
        Cantidad de palabras a la izquierda y derecha de la palabra actual. Default: 2

    Return
    ------
    list
        Lista de diccionarios con las features de cada palabra
    """
    start_pad = ["<BOS>"] * context
    end_pad = ["<EOS>"] * context
    sentence = start_pad + sentence + end_pad
    features = []
    for i in range(len(sentence) - 2 * context):
        aux = []
        for j in range(2 * context + 1):
            aux += [sentence[i + j]]
        features += [aux]
    features = [dict(enumerate(feature)) for feature in features]
    return features

def extract_corpus_features(words_set: list[list[str]], pos_tags_set: list[list[str]]):
    """Extraer las features del corpus

    Parameters
    ----------
    words_set: list[list[str]]
        Lista de listas con las palabras de cada oración
    pos_tags_set: list[list[str]]
        Lista de listas con las etiquetas POS de cada oración

    Return
    ------
    tuple
        Tupla con las features y las etiquetas POS
    """
    X_features = [row for sent in words_set for row in extract_features(sent)]
    y_features = [pos for sent in pos_tags_set for pos in sent]
    return X_features, y_features


# ## Cargando modelo preentrenado de embeddings

# In[19]:


import gensim.downloader as gensim_api
from gensim.models.keyedvectors import KeyedVectors


def get_embeddings(model: KeyedVectors) -> dict[str, torch.FloatTensor]:
    """Obtiene los embeddings de las palabras del modelo

    Parameters
    ----------
    model: KeyedVectors
        Modelo de embeddings

    Return
    ------
    dict[str, torh.FloatTensor]
        Diccionario con las palabras como keys y los embeddings como values
    """
    embeddings = {}
    for word, idx in model.key_to_index.items():
        embeddings[word] = torch.FloatTensor(vectors[idx].copy())
    return embeddings

vectors = gensim_api.load("glove-wiki-gigaword-100")
embeddings = get_embeddings(vectors)


# In[20]:


def extract_pairs(sentence: list[Token], word_key="FORM", pos_key="UPOS"):
    """ Extrae las palabras y sus etiquetas POS

    Parameters
    ----------
    sentence: list[Token]
        Oracion en formato CoNLL
    word_key: str
        Nombre de la columna que contiene la palabra. Default: "FORM"
    pos_key: str
        Nombre de la columna que contiene la etiqueta POS. Default: "UPOS"

    Return
    ------
    tuple
        Tupla con las palabras y sus etiquetas POS
    """
    _input, target = [], []
    for token in sentence:
        _input += [token[word_key]]
        target += [token.get(pos_key, None)]
    return _input, target

def get_raw_corpus(lang: str) -> str:
    """Obtiene el corpus crudo de Universal Dependencies

    Parameters
    ----------
    lang: str
        Idioma del corpus. Puede ser "es" o "en"

    Return
    ------
    str
        Corpus crudo en formato CoNLL
    """
    file_variants = ["train", "test", "dev"]
    result = dict.fromkeys(file_variants)
    DATASETS = {"es": ["UD_Spanish-AnCora", "es_ancora-ud"], "en": ["UD_English-GUM", "en_gum-ud"]}
    repo, file_name = DATASETS[lang]
    for variant in file_variants:
        url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/refs/heads/master/{file_name}-{variant}.conllu"
        r = requests.get(url)
        result[variant] = r.text
    return result

raw_corpus = get_raw_corpus("en")
raw_spanish_corpus = get_raw_corpus("es")


# In[21]:


EMBEDDING_DIM = 100

conll_dict = CoNLLDictorizer()
corpora = {}
for variant in ["train", "test", "dev"]:
    corpora[variant] = conll_dict.transform(raw_corpus[variant])

train_pairs = [extract_pairs(sentence) for sentence in corpora["train"]]
train_sent_words, train_sent_pos = zip(*train_pairs)
train_sent_words_rnn = [list(map(str.lower, sentence)) for sentence in train_sent_words]
corpus_words_rnn = sorted(set([word for sentence in train_sent_words_rnn for word in sentence]))


embeddings_words_rnn = embeddings.keys()
vocabulary = set(corpus_words_rnn + list(embeddings_words_rnn))

embedding_table = torch.randn((len(vocabulary) + 2, EMBEDDING_DIM)) / 10


# In[63]:


embeddings_words_rnn


# In[64]:


corpus_words_rnn


# ## Arquitectura de la RNN

# In[22]:


class RnnModel(nn.Module):
    def __init__(self,
                 embedding_table,
                 hidden_size,
                 num_classes: int,
                 freeze_embeddings: bool = False,
                 num_layers: int=1,
                 bidirectional=False):
        super().__init__()
        embedding_dim = embedding_table.size(dim=-1)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_table,
            freeze=freeze_embeddings,
            padding_idx=0
            )
        self.recurrent = nn.RNN(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        if bidirectional:
            # Dos veces las unidades si es bidireccional
            self.linear = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        rec_out, _ = self.recurrent(embeds)
        logist = self.linear(rec_out)
        return logist


# ## Viendo la pérdida durante el entrenamiento

# In[66]:


history = torch.load("pos_tagger_rnn.history")

def plot_history(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history["accuracy"], label="Accuracy")
    axes[0].plot(history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history["loss"], label="Loss")
    axes[1].plot(history["val_loss"], label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.show()

plot_history(history)


# ## **Obteniendo los embeddings para 100 palabras**
# 
# 
# Para este caso vamos a recordar que el vocabulario que tenemos está conformado por el corpus con el cual va ser entrenado la RNN, sin embargo, al estar usando un modelo preentrenado de embeddings tenemos que considerar en nuestro vocabulario aquellas palabras que no aparecen en nuestro corpus original. Es por esto mismo que el vocabulario se compone tanto del corpus original para entrenar la RNN como palabras de las cuales pertenecen al modelo prenentrenado de embeddings. Además de que también se tiene que considerar como aleatorio, inicialmente, aquellas palabras que no tienen su respectivo embedding en el modelo preentrenado.
# 
# Por ende, lo primero es tal cual es hacer un muestreo de 100 palabras sobre el corpus_words_rnn, para posteriormente extraer sus embeddings usando la capa de embeddings de la RNN. Para este caso se entiende que al hablar de embeddings estáticos es dinámicos, depende directamente de la configuración de la capa de embeddings, entonces tomamos lospesos de la capa de embeddings para poder extraer los correspondientes embeddings usando el indice de la palabra obtenida del corpus_words_rnn.

# In[76]:


X_train_features, y_train_features = extract_corpus_features(train_sent_words, train_sent_pos)
idx2pos = dict(enumerate(set(y_train_features)))
pos2idx = {v: k for k, v in idx2pos.items()}


# In[83]:


X_train_features


# In[34]:


hidden_size = 128           
num_classes = 20            
freeze_embeddings = False   
num_layers = 2
bidirectional = False

model = RnnModel(
    embedding_table=embedding_table,
    hidden_size=hidden_size,
    num_classes=num_classes,
    freeze_embeddings=freeze_embeddings,
    num_layers=num_layers,
    bidirectional=bidirectional
)

model.load_state_dict(torch.load("pos_tagger_rnn_cuda_9.pth", map_location='cuda' if torch.cuda.is_available() else 'cpu'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()


# In[129]:


embedding_weight = model.embedding.weight.data  # (vocab_size, embedding_dim)

vocab_size, embedding_dim = embedding_weight.shape
print(f"Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")
idx2word = dict(enumerate(vocabulary), start=2)
word2idx = {v: k for k, v in idx2word.items()}

# Obtener 100 índices aleatorios dentro del vocabulario
valid_words = [word for window in X_train_features for word in [window[2]] if word.isalpha()]
random_indices = random.sample(range(len(valid_words)), 100)

print("\n\n######### Embeddings de 100 palabras del vocabulario #########\n")
idx2word = {idx: word for word, idx in word2idx.items()}
embedding_seleccionados = []
for idx in random_indices:
    print(f"{idx2word[idx]}: {embedding_weight[idx][:5]}...")
    embedding_seleccionados.append(embedding_weight[idx])


# ## **Aplica un algoritmo de clusterización a las palabras y plotearlas en 2D**
# 
# Básicamente aquí es únicamente hacer el clustering, y para este caso se usó KMeans, además de que se escogió la cantidad de clusters más adecuada por medio del método de silueta. Posteriormente se hizo la reducción de dimensionalidad, y se provee de dos métodos, usando PCA o TSNE.

# In[130]:


from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

def cluster_and_plot(embeddings, reduction_method='pca', k_range=(2, 11), random_state=42):
    """
    Aplica clustering y reducción de dimensionalidad para visualizar embeddings.

    Parámetros:
    - embeddings: np.ndarray o list[list[float]]
        Embeddings en alta dimensión (ej. 100D).
    - reduction_method: str
        Método de reducción a 2D ('pca' o 'tsne').
    - k_range: tuple
        Rango de valores k para evaluar en clustering (ej. (2, 10)).
    - random_state: int
        Semilla para reproducibilidad.

    Retorna:
    - best_k: int
        Número óptimo de clusters.
    """
    embeddings = np.array(embeddings)


    # Encontrar la mejor cantidad de clusters usando el método de silueta, otro método podría ser el de codo.
    silhouette_scores = []
    k_values = list(range(k_range[0], k_range[1]))

    for k in k_values:
        clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric='cosine')
        silhouette_scores.append(score)

    best_k = k_values[silhouette_scores.index(max(silhouette_scores))]

    kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Reducción a 2D con pca o tsne 
    if reduction_method == 'pca':
        reducer = PCA(n_components=2)
    elif reduction_method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=random_state)
    else:
        raise ValueError("Método de reducción no válido. Usa 'pca' o 'tsne'.")

    embeddings_2d = reducer.fit_transform(embeddings)




    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=cluster_labels, cmap='tab10', s=80, edgecolor='k')

    # Añadir leyenda
    plt.legend(*scatter.legend_elements(),
               title="Clusters de embeddings de palabras",
               loc="upper right",
               bbox_to_anchor=(1.15, 1))
    plt.title(f"Clusterización con KMeans (k={best_k}) usando {reduction_method.upper()}")
    plt.xlabel(f"{reduction_method.upper()} 1")
    plt.ylabel(f"{reduction_method.upper()} 2")
    plt.grid(True)
    plt.show()

    return best_k


# In[131]:


best_k = cluster_and_plot(embedding_seleccionados, reduction_method='tsne')


# In[132]:


best_k = cluster_and_plot(embedding_seleccionados, reduction_method='pca')


# ## **Agrega al plot los embeddings de las etiquetas POS**
#  ### Utiliza un marcador que las distinga claramente de las palabras
# 
# Para esto se tiene que modificar la función anterior para poder permitir que también admita los embeddings de las palabras y los embeddings de los POS Tag de cada palabra.
# 
# Sin embargo, aquí es un poco confuso el hecho de dónde sacar los embeddings de las etiquetas POS, ya que en sí lo que tenemos es una capa de embeddings pero solamente para las palabras de entradas. Posteriormente a esto usando los estados ocultos se pasan a una capa densa, o lineal, para después obtener una distribución de probabilidades usando Softmax. Entonces, no hay como tal una parte de la red donde estemos definiendo explícitamente una capa de embeddings para los etiquetas POS. 
# 
# Lo que concluí en esta sección es usar la matriz generada en la capa densa de la RNN, de la cual podríamos interpretar como que cada renglón es un embedding para cada etiqueta POS. Es de esta forma que se van a representar los embeddings de las etiquetas POS.

# In[214]:


pos_embeddings = model.linear.weight.detach().cpu().numpy()

# usamos los indics obtenidos anteriormente random_indices

print("\n\n######### Etiqueta pos de 100 palabras del vocabulario #########\n")
tensor = torch.LongTensor(random_indices).unsqueeze(0).to(device)
outputs = model(tensor)
predictions = torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()

for pred, word in zip(predictions, random_indices):
    print(f'Palabra {idx2word[word]} - etiqueta pos: {idx2pos[pred]}')


# In[238]:


def cluster_and_plot(pos_embeddings, word_embeddings, predictions, idx2word, idx2pos, 
                    reduction_method='pca', random_state=42):
    """
    Visualiza embeddings de palabras agrupadas por POS, con colores distintos por cluster

    Parámetros:
    - pos_embeddings: Embeddings de las etiquetas POS
    - word_embeddings: Embeddings de las palabras
    - predictions: Predicciones POS para cada palabra
    - idx2word: Diccionario de índice a palabra
    - idx2pos: Diccionario de índice a etiqueta POS
    """
    # Conversión a numpy
    word_embeddings = word_embeddings.detach().cpu().numpy() if isinstance(word_embeddings, torch.Tensor) else word_embeddings
    pos_embeddings = pos_embeddings.detach().cpu().numpy() if isinstance(pos_embeddings, torch.Tensor) else pos_embeddings

    # Número único de etiquetas POS
    unique_pos = np.unique(predictions)
    n_clusters = len(unique_pos)

    # Reducción dimensional
    if reduction_method == 'pca':
        reducer_word = PCA(n_components=2, random_state=random_state)
        reducer_pos = PCA(n_components=2, random_state=random_state)
    elif reduction_method == 'tsne':
        min_samples = min(word_embeddings.shape[0], pos_embeddings.shape[0])
        perplexity = min(30, min_samples - 1)  # t-SNE requiere perplexity < n_samples
        reducer_word = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=random_state)
        reducer_pos = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=random_state)
    else:
        raise ValueError("Método de reducción no válido. Usa 'pca' o 'tsne'.")

    word_points = reducer_word.fit_transform(word_embeddings)
    pos_points = reducer_pos.fit_transform(pos_embeddings)

    # Crear figura
    plt.figure(figsize=(16, 12))

    for i, pos_idx in enumerate(unique_pos):
        mask = predictions == pos_idx
        pos_label = idx2pos[pos_idx]

        plt.scatter(word_points[mask, 0], word_points[mask, 1], 
                   color=plt.cm.tab20(i), 
                   s=100, alpha=0.7, 
                   label=f'{pos_label} ({np.sum(mask)} palabras)')

    # Embeddings POS con marcadores especiales
    for i, pos_idx in enumerate(unique_pos):
        pos_label = str(idx2pos[pos_idx])
        safe_label = pos_label.replace('_', '-')
        if safe_label == '-' and safe_label == 'X':
            continue
        plt.scatter(pos_points[pos_idx, 0], pos_points[pos_idx, 1],
                   color='black', marker=f'${safe_label}$',
                   s=800, alpha=1.0, edgecolors='white', linewidths=1.5)

    # Etiquetas a algunas palabras representativas
    for pos_idx in unique_pos:
        mask = predictions == pos_idx
        if np.sum(mask) > 0:
            centroid = np.mean(word_points[mask], axis=0)
            distances = np.linalg.norm(word_points[mask] - centroid, axis=1)
            closest_idx = np.where(mask)[0][np.argmin(distances)]

            plt.annotate(idx2word[closest_idx], 
                        (word_points[closest_idx, 0], word_points[closest_idx, 1]),
                        fontsize=9, alpha=0.9, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8))

    plt.title(f'Distribución de Embeddings por Categoría POS\n{len(unique_pos)} clusters', fontsize=14)
    plt.xlabel(f'Componente {reduction_method.upper()} 1', fontsize=12)
    plt.ylabel(f'Componente {reduction_method.upper()} 2', fontsize=12)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# In[ ]:


def cluster_and_plot(pos_embeddings, word_embeddings, predictions, idx2word, idx2pos, 
                    reduction_method='pca', random_state=42):
    """
    Visualiza embeddings de palabras y POS por separado

    Parámetros:
    - pos_embeddings: np.ndarray o torch.Tensor, embeddings de POS [num_pos_tags, pos_dim]
    - word_embeddings: np.ndarray o torch.Tensor, embeddings de palabras [num_words, word_dim]
    - predictions: np.ndarray, predicciones POS para cada palabra
    - idx2word: dict, mapeo de índice a palabra
    - idx2pos: dict, mapeo de índice a etiqueta POS
    """
    # Convertir a numpy arrays si son tensores
    if isinstance(word_embeddings, torch.Tensor):
        word_embeddings = word_embeddings.detach().cpu().numpy()
    if isinstance(pos_embeddings, torch.Tensor):
        pos_embeddings = pos_embeddings.detach().cpu().numpy()

    # Reducción dimensional separada para palabras y POS
    if reduction_method == 'pca':
        reducer_word = PCA(n_components=2, random_state=random_state)
        reducer_pos = PCA(n_components=2, random_state=random_state)
    elif reduction_method == 'tsne':
        reducer_word = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=random_state)
        reducer_pos = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=random_state)
    else:
        raise ValueError("Método de reducción no válido. Usa 'pca' o 'tsne'.")

    # Aplicar reducción dimensional por separado
    word_points = reducer_word.fit_transform(word_embeddings)
    pos_points = reducer_pos.fit_transform(pos_embeddings)

    # Obtener nombres de etiquetas POS
    pos_tags = [str(idx2pos[i]).replace('X','x') for i in predictions]

    # Crear figura
    plt.figure(figsize=(14, 10))

    # Palabras coloreadas por POS predicho
    scatter = plt.scatter(word_points[:, 0], word_points[:, 1], 
                         c=predictions, cmap='tab20', s=80, 
                         alpha=0.7, marker='o', label='Palabras')

    # Etiquetas POS (con marcadores distintos)
    for i, pos in enumerate(predictions):
        plt.scatter(pos_points[pos, 0], pos_points[pos, 1], 
                   c='black', marker=f'${str(idx2pos[pos]).replace('_','-')}$', s=1000, 
                   alpha=1.0, label='Etiquetas POS' if i == 0 else "")

    # Añadir etiquetas a algunas palabras clave
    labeled_indices = np.linspace(0, len(word_embeddings)-1, min(20, len(word_embeddings)), dtype=int)
    for i in labeled_indices:
        plt.annotate(idx2word[i], (word_points[i, 0], word_points[i, 1]), 
                    fontsize=8, alpha=0.8)

    # Leyenda personalizada
    unique_preds = np.unique(predictions)
    plt.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i), markersize=10) 
                for i in range(len(unique_preds))],
               [idx2pos[p] for p in unique_preds], 
               title="POS Predicho", loc="upper right")

    plt.title("Visualización de Embeddings (proyectados por separado)")
    plt.xlabel(f"Componente {reduction_method.upper()} 1")
    plt.ylabel(f"Componente {reduction_method.upper()} 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# In[240]:


# embeddings
pos_embeddings = model.linear.weight  # shape [num_pos_tags, hidden_size]
word_embeddings = model.embedding.weight[random_indices]  # shape [100, embedding_dim]

# predicciones
tensor = torch.LongTensor(random_indices).unsqueeze(0).to(device)
outputs = model(tensor)
predictions = torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()

cluster_and_plot(pos_embeddings, word_embeddings, predictions, idx2word, idx2pos, reduction_method='pca')


# In[279]:


def cluster_and_plot(embeddings, pos_embeddings, predictions, pos_labels, 
                    reduction_method='pca', random_state=42):
    """
    Visualiza clusters de palabras y embeddings POS con esquema de colores distinto pero consistente

    Parámetros:
    - embeddings: Tensor de embeddings de palabras
    - pos_embeddings: Tensor de embeddings POS
    - predictions: Array con etiquetas POS predichas para cada palabra
    - pos_labels: Lista de nombres de etiquetas POS
    - reduction_method: 'pca' o 'tsne'
    - random_state: Semilla aleatoria
    """
    # Convertir a numpy
    word_emb = embeddings.detach().cpu().numpy()
    pos_emb = pos_embeddings.detach().cpu().numpy()


    # Reducción dimensional
    if reduction_method == 'pca':
        reducer_word = PCA(n_components=2, random_state=random_state)
        reducer_pos = PCA(n_components=2, random_state=random_state)
    elif reduction_method == 'tsne':
        min_samples = min(word_embeddings.shape[0], pos_embeddings.shape[0])
        perplexity = min(30, min_samples - 1)  # t-SNE requiere perplexity < n_samples
        reducer_word = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=random_state)
        reducer_pos = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=random_state)
    else:
        raise ValueError("Método de reducción no válido. Usa 'pca' o 'tsne'.")

    word_2d = reducer_word.fit_transform(word_embeddings.detach().numpy())
    pos_2d = reducer_pos.fit_transform(pos_embeddings.detach().numpy())

    # clustering
    n_clusters = len(np.unique(predictions))
    n_pos = len(pos_labels)

    word_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    pos_colors = plt.cm.tab20b(np.linspace(0, 1, n_pos))


    plt.figure(figsize=(14, 10))

    # embeddings
    for cluster in range(n_clusters):
        mask = predictions == cluster
        plt.scatter(word_2d[mask, 0], word_2d[mask, 1],
                   color=word_colors[cluster],
                   s=50, alpha=0.6, edgecolor='k',
                   label=f'Cluster {cluster}')

    # etiquetas pos
    markers = ['*', 'D', 'P', 'X', 's', '^', 'v', '<', '>', 'p']
    for pos_idx in range(n_pos):
        plt.scatter(pos_2d[pos_idx, 0], pos_2d[pos_idx, 1],
                   marker=markers[pos_idx % len(markers)],
                   s=300, color=pos_colors[pos_idx],
                   edgecolor='black', linewidth=1,
                   label=f'{pos_labels[pos_idx]} (POS)')


    word_legend = plt.legend(title="Clusters de palabras",
                           bbox_to_anchor=(1.05, 1), 
                           loc='lower left')
    #plt.gca().add_artist(word_legend)

    plt.legend(title="Etiquetas POS",
                          bbox_to_anchor=(1.05, 0.7),
                          loc='lower left')

    plt.title(f"Clusters de Palabras y Embeddings POS\n({reduction_method.upper()})")
    plt.xlabel(f"Componente {reduction_method.upper()} 1")
    plt.ylabel(f"Componente {reduction_method.upper()} 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# In[280]:


cluster_and_plot(
    embeddings=word_embeddings,
    pos_embeddings=pos_embeddings,
    predictions=predictions,
    pos_labels=list(idx2pos.values()),  # tus etiquetas
    reduction_method='pca'
)


# Se peude notar que parece existir cierta relación en el espacio, pero sí sigue mostrándose disperso los diferentes items que pertenece a cada cluster. De manera que podemos notar que en efecto parecen estar muy dispersos las palabras que supuestamente tienen la misma etiqueta POS.
# 
# Es algo curioso, a pesar de que tenemos que la pérdida baja bien en el entrenamiento, además de un accuracy decente, tenemos estas representaciones un poco raras. Muy posiblemente se debe a la alta dimensionalidad, pero en sí lo que podemos ver de los resultados de entrenamiento y validación es que en efecto el modelo sí es capaz de asociar bien las etiquetas POS para las palabras.
