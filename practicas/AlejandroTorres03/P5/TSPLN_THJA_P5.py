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

# %% [markdown] id="lgdiSBZEI2fa"
# # Práctica 5: Tech evolution. Caso POS Tagging
#
# - Obten los embeddings de 100 palabras al azar del modelo RNN visto en clase
#   - Pueden ser los embeddings estáticos o los dinámicos del modelo
# - Aplica un algoritmo de clusterización a las palabras y plotearlas en 2D
#   - Aplica algun color para los diferentes clusters
# - Agrega al plot los embeddings de las etiquetas POS
#   - Utiliza un marcador que las distinga claramente de las palabras
# - Realiza una conclusión sobre los resultados observados
#
# ### Extra: 0.5pt
#
# - Implementa una red *Long short-term memory units (LSTM)* para la tarea de etiquetado POS
# - Reporta el accuracy y comparalo con los resultados de la RNN simple
# - Realiza un comentario sobre como impacta la arquitectura LSTM sobre el resultado obtenido

# %% [markdown] id="-C2hCJHYLCfk"
# #Funciones Creadas Durante La Clase

# %% colab={"base_uri": "https://localhost:8080/"} id="dm56xAUYLCNg" outputId="ecc8e91a-633e-4e70-efdf-ad4daab47291"
# !pip install numpy==1.26.4
# !pip install -U gensim
# !pip install utils
#Reiniciar entorno!

# %% id="f1pXK0cBLRWO"
from rich import print as rprint
import requests
import gensim.downloader as gensim_api
from gensim.models.keyedvectors import KeyedVectors
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# %% id="RZQQo1IuLUTp"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% id="KE0_x8kOLpkH"
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


# %% id="C1xuZdhpMp8E"
raw_corpus = get_raw_corpus("en")
raw_spanish_corpus = get_raw_corpus("es")


# %% id="8tm-LOwTMrxO"
class Token(dict):
    """Modela cada renglon de un corpus en formato CoNLL
    """
    pass


# %% colab={"base_uri": "https://localhost:8080/", "height": 129} id="40NTx_H1MwYc" outputId="a4043a69-2fa9-4b25-de2e-335493ff8b34"
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

# %% id="-JkKFSpyMxt3"
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


# %% id="Vhzj1DZ4M1Th"
conll_dict = CoNLLDictorizer()

# %% id="sXjoywVSM4Qk"
corpora = {}
for variant in ["train", "test", "dev"]:
    corpora[variant] = conll_dict.transform(raw_corpus[variant])

# %% id="gZzGY954M5z5"
spanish_corpora = {}
for variant in ["train", "test", "dev"]:
    spanish_corpora[variant] = conll_dict.transform(raw_spanish_corpus[variant])


# %% id="aJYl2fFSM7eu"
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


# %% id="ihkkBxa4M9uu"
train_pairs = [extract_pairs(sentence) for sentence in corpora["train"]]
val_pairs = [extract_pairs(sentence) for sentence in corpora["dev"]]
test_pairs = [extract_pairs(sentence) for sentence in corpora["test"]]

# %% id="nstD-7qMM_O9"
train_sent_words, train_sent_pos = zip(*train_pairs)
val_sent_words, val_sent_pos = zip(*val_pairs)
test_sent_words, test_sent_pos = zip(*test_pairs)

# %% colab={"base_uri": "https://localhost:8080/"} id="4NOIfnJNNAeO" outputId="de4d1cfa-9a83-4f98-d066-4515a71ccee3"
vectors = gensim_api.load("glove-wiki-gigaword-100")


# %% id="9-UOSTPGNCCc"
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


# %% id="HNVRf7qMNOfz"
embeddings = get_embeddings(vectors)

# %% colab={"base_uri": "https://localhost:8080/", "height": 49} id="BZOUadqsNQQS" outputId="58dcabec-8dd5-49f6-b0f1-fe3facc2bc0d"
# Extracting the training set
rprint(train_sent_words[0])
rprint(train_sent_pos[0])

# %% id="4QpUWRm4NSp1"
# Todas las palabras a minusculas para que sea compatible con GLoVe
train_sent_words_rnn = [list(map(str.lower, sentence)) for sentence in train_sent_words]
val_sent_words_rnn = [list(map(str.lower, sentence)) for sentence in val_sent_words]
test_sent_words_rnn = [list(map(str.lower, sentence)) for sentence in test_sent_words]

# %% [markdown] id="SfSlgM8pNWaO"
# ##Indices

# %% id="jmARxzTDNU-7"
corpus_words_rnn = sorted(set([word for sentence in train_sent_words_rnn for word in sentence]))
pos_list_rnn = sorted(set([pos for sentence in train_sent_pos for pos in sentence]))

# %% id="cp2leSwONY-a"
embeddings_words_rnn = embeddings.keys()
vocabulary = set(corpus_words_rnn + list(embeddings_words_rnn))

# %% id="yeAAypcmNaaC"
# Start on 2 because id 0 will be pad simbol and 1 will be UNK
idx2word = dict(enumerate(vocabulary), start=2)
idx2pos = dict(enumerate(pos_list_rnn), start=1)

word2idx = {v: k for k, v in idx2word.items()}
pos2idx = {v: k for k, v in idx2pos.items()}


# %% id="yvOp3OgqNbd5"
def to_index(corpus: list[list[str]], word2idx: dict[str, int], unk_id: int = 1) -> torch.LongTensor:
    indexes = []
    for sent in corpus:
        sent_indexes = torch.LongTensor(
            list(map(lambda word: word2idx.get(word, unk_id), sent))
        )
        indexes += [sent_indexes]
    return indexes


# %% id="34VHwltnNcu4"
t = to_index(train_sent_words_rnn[:2], word2idx)

# %% colab={"base_uri": "https://localhost:8080/"} id="LLKKP5sYNd-4" outputId="23594a68-7102-44a6-cf48-25268461a100"
for sent in t:
    for word in sent:
        print(idx2word[int(word)])

# %% id="h4ReyDVBNfJh"
X_train_idx_rnn = to_index(train_sent_words_rnn, word2idx)
Y_train_idx_rnn = to_index(train_sent_pos, pos2idx)

# %% id="kFc0U-uWNhqE"
X_val_idx_rnn = to_index(val_sent_words_rnn, word2idx)
Y_val_idx_rnn = to_index(val_sent_pos, pos2idx)

# %% id="krcWS20SNil6"
X_test_idx_rnn = to_index(test_sent_words_rnn, word2idx)
Y_test_idx_rnn = to_index(test_sent_pos, pos2idx)

# %% [markdown] id="jOGyadytNkfk"
# ##Padding

# %% colab={"base_uri": "https://localhost:8080/"} id="OH1VW0JqNjmW" outputId="b7cafae7-7de1-4ec2-93d8-e7c0882be032"
pad_sequence(X_train_idx_rnn[41:43], batch_first=True, padding_value=0)

# %% id="8PLf8GXZNsZy"
X_train_rnn = pad_sequence(X_train_idx_rnn, batch_first=True, padding_value=0)
Y_train_rnn = pad_sequence(Y_train_idx_rnn, batch_first=True, padding_value=0)

# %% id="MSXsWr6MN2qp"
X_val_rnn = pad_sequence(X_val_idx_rnn, batch_first=True, padding_value=0)
Y_val_rnn = pad_sequence(Y_val_idx_rnn, batch_first=True, padding_value=0)

# %% id="giCwXS2IN32R"
X_test_rnn = pad_sequence(X_test_idx_rnn, batch_first=True, padding_value=0)
Y_test_rnn = pad_sequence(Y_test_idx_rnn, batch_first=True, padding_value=0)

# %% [markdown] id="mg4fXtkUN-8b"
# ##Embeddings para RNN

# %% id="ZKQbbojhN5EI"
EMBEDDING_DIM = 100

# %% id="LKpCA1AvOB7C"
embedding_table = torch.randn((len(vocabulary) + 2, EMBEDDING_DIM)) / 10

# %% id="x0WfC5hiODDC"
for word in vocabulary:
    if word in embeddings:
        embedding_table[word2idx[word]] = embeddings[word]


# %% [markdown] id="lk03lI1kOEtn"
# #La RNN con pytorch

# %% id="6qfoDC4KOEPX"
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


# %% id="GCX3VphKOJHB"
rnn_model = RnnModel(
    embedding_table,
    hidden_size=128,
    num_classes=len(pos2idx) + 1,
    freeze_embeddings=False,
    num_layers=2,
    bidirectional=True
).to(device)

# %% id="nJKBvY35OKUz"
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.NAdam(rnn_model.parameters(), lr=0.005)

# %% [markdown] id="cA2oRcGsON71"
# ###Trainingloop

# %% id="seR3HoOLOL5F"
EPOCHS = 10

# %% id="vWW4LxPbOSE-"
train_dataset_rnn = TensorDataset(X_train_rnn, Y_train_rnn)
train_dataloader_rnn = DataLoader(train_dataset_rnn, batch_size=512, shuffle=True)

# %% id="8v9wCZDTOTIu"
val_dataset_rnn = TensorDataset(X_val_rnn, Y_val_rnn)
val_dataloader_rnn = DataLoader(val_dataset_rnn, batch_size=2048, shuffle=False)

# %% id="RHiRENkGOUMV"
test_dataset_rnn = TensorDataset(X_test_rnn, Y_test_rnn)
test_dataloader_rnn = DataLoader(test_dataset_rnn, batch_size=2048, shuffle=False)


# %% id="1LjDn9GsOVia"
def evaluate_rnn(model: nn.Module, loss_fn: nn.Module, dataloader: DataLoader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        accuracy = 0
        t_words = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch_pred = model(X_batch)
            current_loss = loss_fn(
                y_batch_pred.reshape(-1, y_batch_pred.size(dim=-1)),
                y_batch.reshape(-1)
            )
            n_words = torch.sum(y_batch > 0).item()
            t_words += n_words
            total_loss += n_words + current_loss.item()
            accuracy += torch.mul(
                torch.argmax(y_batch_pred, dim=-1) == y_batch,
                y_batch > 0).sum().item()
        return total_loss / t_words, accuracy / t_words


# %% id="gPM6__eCOXCF"
import os

# Crear el directorio si no existe
MODELS_PATH = "./models"
os.makedirs(MODELS_PATH, exist_ok=True)


# %% colab={"base_uri": "https://localhost:8080/", "height": 369, "referenced_widgets": ["4bbd88fa3eb14205bae07746d8f35698", "911f67ea2acf42dcb3c217c05a9b2ae3", "5690b963d5654ad9bcdb80417be98be4", "5573060cd9da4e9abd60d5b90f209b98", "74ca773e21f64f35b54f64bb1c9bedce", "67a60d403a5042329860eee45c0b5b07", "fa28f6c91b364be09cad940840eaf6e5", "856947ccb7474d42bb35adeb33b5676c", "27666ca0e5b64c1fb1dd93373b6c9709", "2645eb8bf3f6420ebd2b0f3181bcbe13", "139cbd010e8d414f973e7744455f04f1", "799dd7bdcd414fa283d1266b95044f4e", "ef4ab65b634c477b833ae6fd524fe28c", "cab2fc862910463ea5cb10f4c92ba421", "6e8f0b28881046f7881e95777c0591c6", "8f8b78464baa4488ab47a4df6c8985f8", "be47d00db0564f629cac7f13b58fd85c", "7e199f9ad7b9407982166315f53b5623", "0e8e9791dd714cce999cee3c04a1061b", "647004ed58fe4b29bfc649961ffcbbd7", "c13f20567065466fa0d6250f882ba34b", "a2fcdff1105f4ce6b4f3679dc7ded477", "dd7d3fe16786437c95f30c7f3ee9cd6f", "2035d29511f34e93b15a775a3a17399a", "270afd9cbe4b462d8857c91ebdca1b64", "31a99a388d0d47f6a65ae3fbdfcb35b0", "25a038e9c3c74d4198dfe8821348f26e", "29de34392e3f49c8a2da9236f73d9f60", "c6fa0efb52bf438982db91f341f331fc", "1b605ef7946945f1aeaf664a02a68394", "3bd583e0164740c8b046a6ed8068d27b", "59324c364bda44a2a5329df097b38421", "83a8edfe9a7843e182acbaa7dc3f7957", "58f107350bc8445ba3c07a0638d07e8a", "867b2d9e98374172a48c92434c4cfd41", "319369b95fab49f8b6d7b944d42f771d", "bd0ee1b05d114a5a96e402f681c7a05c", "ddd0f6fbfdf840edb21421ebb31b9094", "fa7db0af553547ab8620cb877f5ad81a", "9bfaf4c598464043a4e1a404845879d5", "31d17c22fdc24c69a642f08bc49ee6f6", "568f94aae91842f49171cb676c316358", "6da597a5909b42daad4fe1a5fd0e7301", "e886bbf735574510b5ab274af32036f2", "1142ec79a40c4b8bb7d5cfdca2d39d33", "c419261ce9ce48d7b460076556b5edf1", "e37d62fc3a48464f859ff0cc460dde0e", "7c9dbe8037fa4bf08351ff8ebf724d5d", "ee2421572b074853ae87f26b6f913709", "8c2a0c116e2445e08564a473d553df05", "4dcdfff49caf4b38be0eca48d98dca73", "ea2a40be12d84538ad2e2f4ec90664c4", "3122013803ee430092317fa886e3f920", "3ad0497769484f12ab6f4c3fee5cde00", "2ce4ac8951a0462a82d8408147515ff0", "de8487af4d0543bb952694e721767523", "5557ec055a8e455ca27bf0afec44e952", "3271fd48ce484cef8ebd310e2ce43c8c", "baa9723d5ab2405f8a6002963c2d08ed", "cda1c811be5f40f1822319a6341df5d4", "4618fd2850cd49bd93c613be555236a5", "71c5837c20714eec95b581bf038ef05d", "7bd95ad63c2e418cbadaf7b4c2b10332", "8fe162424f6f4805b3e518ed3c35e69b", "6461b3896b904e8ca0967c83c63b6e4e", "eca17ffd2a364f0e83565abbd14773c7", "2e71604da86541c8a6500a61b766fbd3", "7905d1ff5d2143149cf8d8c061149e3f", "9d6b3f959f684855aae079bfd27fa0c3", "e2dbae06decf4ba3b784263a4e9aca9f", "3a2d3634553b46ae9ba40c050060a6c5", "fd1540c800414dadb736b6e31dd1beaa", "ceb4988d904a4ea59a311b6362b2bc1d", "68f5d20a963c44b2bb76a591a11a60f7", "3becfd60c1944c1bba2bca06ccca5251", "2f67fd69370848c9b44c112480aa3718", "7caea4a482fc490ca835f3f1fc8e5eae", "bdda2a977f6e437ab0c7f280fbdc5599", "0632b906e40943ea8f0256d3a9206787", "6fa4f7ea9ea0412f9977c9b0c04176fc", "2711abb136b74377b3ed8107a54e5b6a", "9ad6906088734333a561d1604c5711f2", "d9db5af6a32046e19f9f938b7f11048b", "04099fd1a96344428ceeeec174a8248a", "6c582eb6e3ed41a198e49d37ecc90e63", "d2d6e3b2fc9e4237ab6f661868c2701c", "0df2e1436f3f4b23a666dcb30856e88f", "562ebd7cf77d44a59b4304f8cff5a58e", "9e35aa0b80744bbb902ec359a050628b", "bfe33334e0ed4eb3bc3134d36745e5fe", "e185ee857e994268b12c64fd4e1c20f0", "0ebfc34824c744a2bd22c36b04012738", "c90a1b4baaac4a38b2b3b3a51bba1666", "102892d8bb3b40eaa26468472f7f2b0a", "398e633ef08f46799da73d94a03abff6", "7a70375993f6423fa1a741ab4b8e907a", "8283b1d46c01409da165f2e3747d843f", "d21d2f00230a41809775797a8eba777d", "9076c45934e2428e95bfd507f9c3b575", "b5232d0a94e44ee8ad530c4a8d1340e3", "16bbfde63d3e473f8540dd84c47668d9", "21a6ec4fa91245fd9ad93b0e64786668", "ecdd5bac4e024d778d62451e373e5ab6", "a05bcbc5b318478fb9c68f23ec5b480f", "1968db6644b643cba80a464730b599ee", "8e0a28c5e4b4488a88899c7f143130d3", "4b9f09ed438e46b498d41bee465ad78b", "680303c05de54b2083e7e105dc4529c2", "9341c2912ea04ff18666cb0106370f2e", "6df7c6f977c6490485a3a7ac985e5459", "123a3d6753c14710a2ea1cfd9fd77a0f", "4bcfed83d82049d9a06838b161fb7114", "e572eca668324278ae5f8128791eefae", "1f99ff15f3704e77bc04811c5b04f25c", "c8c808f51e004425858dfae505cd9624", "1ec5cc85525241849341f86b182152d4", "c06badbf685046e4be2fd582f0cc6b0c", "c179396898ea4f6db2f48b48ce078cc3", "a0d385f9f9714d03be27426a12a5ef09", "0f15863730cf40a8ab9c9753dc09ba88", "2d6885939b48462a899cc10fa76de5ac"]} id="DMIIUPAaOYdB" outputId="b1075903-0445-4423-b145-4aa5360384a1"
rnn_history = {"accuracy": [], "loss": [], "val_loss": [], "val_accuracy": []}
for epoch in trange(EPOCHS):
    train_loss, train_acc, t_words = 0, 0, 0
    rnn_model.train()
    for X_batch, y_batch in tqdm(train_dataloader_rnn):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_batch_pred = rnn_model(X_batch)
        loss = loss_fn(
            y_batch_pred.reshape(-1, y_batch_pred.size(dim=-1)),
            y_batch.reshape(-1)
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            n_words = torch.sum(y_batch > 0).item()
            t_words += n_words
            train_loss += n_words * loss.item()
            train_acc += torch.mul(
                torch.argmax(y_batch_pred, dim=-1) == y_batch,
                y_batch > 0).sum().item()

    rnn_model.eval()
    with torch.no_grad():
        rnn_history["accuracy"].append(train_acc / t_words)
        rnn_history["loss"].append(train_loss / t_words)
        val_loss, val_acc = evaluate_rnn(rnn_model, loss_fn, val_dataloader_rnn)
        rnn_history["val_loss"].append(val_loss)
        rnn_history["val_accuracy"].append(val_acc)
    torch.save(rnn_model.state_dict(), f"{MODELS_PATH}/pos_tagger_rnn_{device}_{epoch}.pth")
torch.save(rnn_history, f"{MODELS_PATH}/pos_tagger_rnn.history")


# %% id="AobG1c-7OaOD"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 422} id="NOYS0IohPsnL" outputId="6563b0a7-a9b8-40e5-e26d-635778e56195"
plot_history(rnn_history)

# %% colab={"base_uri": "https://localhost:8080/", "height": 33} id="C1rE_supPvMK" outputId="cb6617b8-aa42-41bc-8378-204419c3c20d"
_, accuracy_rnn = evaluate_rnn(rnn_model, loss_fn, test_dataloader_rnn)
rprint(f"Accuracy (RNN): {accuracy_rnn * 100:.2f}%")

# %% [markdown] id="ciOR-LGBQO8R"
# #Primer ejercicio

# %% [markdown] id="P9NTAg37QRYz"
# - Obten los embeddings de 100 palabras al azar del modelo RNN visto en clase
#   - Pueden ser los embeddings estáticos o los dinámicos del modelo

# %% [markdown] id="nTu_h49hQO5z"
# ##Embeddings estáticos

# %% colab={"base_uri": "https://localhost:8080/"} id="H7GjaGtMP3xN" outputId="361dfbd6-9042-493e-a9a2-458dea9da23e"
import random

# Seleccionar 100 palabras aleatorias del vocabulario
random_words = random.sample(list(word2idx.keys()), 100)

# Obtener los índices de estas palabras
word_indices = torch.tensor([word2idx[word] for word in random_words]).to(device)

# Obtener los embeddings estáticos
with torch.no_grad():
    static_embeddings = rnn_model.embedding(word_indices).cpu().numpy()

# Mostrar algunas palabras y la forma de los embeddings
print(f"Embeddings estáticos para {len(random_words)} palabras")
print(f"Forma de los embeddings: {static_embeddings.shape}")
print("\nEjemplos:")
for i in range(99):
    print(f"Palabra: {random_words[i]}")
    print(f"Embedding (primeros 5 valores): {static_embeddings[i][:5]}")

# %% [markdown] id="b5EZ0L-6R3vD"
# #Segundo Ejercicio

# %% [markdown] id="ynxF2SNrR5V-"
# - Aplica un algoritmo de clusterización a las palabras y plotearlas en 2D
#   - Aplica algun color para los diferentes clusters

# %% colab={"base_uri": "https://localhost:8080/", "height": 790} id="pH7xKktHRiKS" outputId="54e1b8c4-06af-4b88-e495-3ab6817281e8"

# 3. Reducir dimensionalidad a 2D con PCA (para visualización)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(static_embeddings)

# 4. Aplicar K-means para clusterizar las palabras en 5 grupos
n_clusters = 5  # Puedes ajustar este valor
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(static_embeddings)

# 5. Graficar con colores por cluster
plt.figure(figsize=(12, 8))
colors = plt.cm.get_cmap('tab10', n_clusters)  # Paleta de colores

# Scatter plot con colores según el cluster
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1],
    c=clusters, cmap='tab10', alpha=0.6, s=50
)

# Etiquetar palabras (opcional)
for i, word in enumerate(random_words):
    if i % 5 == 0:
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, alpha=0.8)

# Colorbar con clusters enteros
cbar = plt.colorbar(scatter, label='Cluster',
                    ticks=np.arange(n_clusters),
                    boundaries=np.arange(n_clusters+1)-0.5)
cbar.set_ticklabels([f'Cluster {i+1}' for i in range(n_clusters)])

plt.title("Clustering de palabras (K-means + PCA)")
plt.xlabel("Componente PCA 1")
plt.ylabel("Componente PCA 2")
plt.grid(alpha=0.2)
plt.show()

# %% [markdown] id="GIOBX_ABUwry"
# #Tercer Ejercicio

# %% [markdown] id="qI0HjS0AU0Ef"
# - Agrega al plot los embeddings de las etiquetas POS
#   - Utiliza un marcador que las distinga claramente de las palabras

# %% colab={"base_uri": "https://localhost:8080/", "height": 659} id="1LnO7nZxX-GV" outputId="48de3860-3bcf-4e23-a018-34b34e4579a3"
# Obtener embeddings POS desde la capa lineal del modelo
pos_embeddings_dynamic = {
    pos: rnn_model.linear.weight.data.cpu().numpy()[int(pos_id)-1]  # -1 porque el padding es 0
    for pos, pos_id in pos2idx.items()
    if pos not in ['_', 'X', '<pad>'] and str(pos_id).isdigit()  # Filtramos tags especiales
}


# Reducción de dimensionalidad para POS
pca_pos = PCA(n_components=2)
pos_embs = np.array(list(pos_embeddings_dynamic.values()))
pos_2d = pca_pos.fit_transform(pos_embs)
pos_tags = list(pos_embeddings_dynamic.keys())

# Configuración del gráfico
plt.figure(figsize=(14, 8))

# 1. Graficar palabras (clusters)
scatter_words = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1],
    c=clusters, cmap='tab10', alpha=0.6, s=50
)

# 2. Graficar POS con etiquetas individuales en la leyenda
pos_markers = []
for i, pos in enumerate(pos_tags):
    # Graficar cada POS
    marker = plt.scatter(
        pos_2d[i, 0], pos_2d[i, 1],
        marker='X', s=150, edgecolor='black', linewidth=1.5,
        color='darkred', zorder=4,
        label=pos  # Etiqueta individual para la leyenda
    )
    plt.annotate(pos, (pos_2d[i, 0], pos_2d[i, 1]),
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    pos_markers.append(marker)

# 3. Colorbar con valores enteros
cbar = plt.colorbar(
    scatter_words,
    label='Cluster de palabras',
    ticks=np.arange(n_clusters),
    boundaries=np.arange(n_clusters+1)-0.5
)
cbar.set_ticklabels([f'Cluster {i+1}' for i in range(n_clusters)])

# 4. Leyenda para POS (todas las etiquetas)
plt.legend(
    handles=pos_markers,
    labels=pos_tags,
    title="Etiquetas POS",
    loc='upper right',
    framealpha=1,
    handletextpad=0.5
)

# Ajustes finales
plt.title("Clustering de palabras con etiquetas POS", fontsize=14)
plt.xlabel("Componente PCA 1")
plt.ylabel("Componente PCA 2")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %% [markdown] id="x38iyMPTlqGq"
# En base a lo observado podemos decir que el modelo RNN logra agrupar palabras que tienen cierta relación entre sí, ya sea por su significado o por cómo se usan en el lenguaje. Al aplicar clustering y reducir la dimensión con PCA, se forman grupos bastante coherentes, aunque no siempre corresponden a una sola categoría gramatical. Al agregar las etiquetas POS, se nota que están distribuidas en distintas zonas del espacio, lo que indica que el modelo también está captando diferencias en la función de las palabras. Algunos puntos quedan más aislados, probablemente porque son palabras menos comunes o con un uso más específico. En general, los resultados son buenos considerando que se usaron embeddings estáticos, aunque usar embeddings dinámicos podría ayudar a reflejar mejor el contexto en el que se usan las palabras.
#
