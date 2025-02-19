# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="76b3a996-772a-4be8-a8eb-f1e9ae67d03e"
# # 1. Niveles Lingüísticos

# + [markdown] id="615a09ab-2b52-440a-a4dc-fd8982c3c0e7"
# ## Objetivos

# + [markdown] id="f034458c-9cb0-4966-a203-3145074c3fca"
# - Trabajar tareas a diferentes niveles lingüísticos (Fonético, Morfólogico, Sintáctico)
# - Manipularan y recuperará información de datasets disponibles en Github para resolver tareas de NLP
# - Comparar enfoques basados en reglas y estadísticos para el análisis morfológico

# + [markdown] id="3c169487-91d2-4afb-a12a-849c26a5be86"
# ## Fonética y Fonología

# + [markdown] id="d0647e1e-a8c5-418f-81c7-31d2e86c88a4"
# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/IPA_chart_2020.svg/660px-IPA_chart_2020.svg.png"></center

# + colab={"base_uri": "https://localhost:8080/", "height": 623} executionInfo={"elapsed": 37, "status": "ok", "timestamp": 1738771093307, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="ecb34504-4988-4fca-82f1-824ba96a0909" outputId="faff1ea5-3eff-4ec5-fb77-3a2e1dc01815"
# %%HTML
<center><iframe width='901' height='600' src='https://www.youtube.com/embed/DcNMCB-Gsn8?controls=1'></iframe></center

# + colab={"base_uri": "https://localhost:8080/", "height": 623} executionInfo={"elapsed": 39, "status": "ok", "timestamp": 1738775748361, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="1HPZ7Xpt9kgF" outputId="f9a1f3f7-a00e-4778-f18d-f3938ebe2289"
# %%HTML
<center><iframe width='900' height='600' src='https://www.youtube.com/embed/74nnLh0Vdcc?controls=1'></iframe></center>

# + [markdown] id="aa915e8e-038e-4de6-8956-f6221b1d8490"
# ### International Phonetic Alphabet (IPA)

# + [markdown] id="09b4f076-b23b-46a8-9101-e37d79d374c8"
# - Las lenguas naturales tienen muchos sonidos diferentes por lo que necesitamos una forma de describirlos independientemente de las lenguas
# - IPA es una representación escrita de los [sonidos](https://www.ipachart.com/) del [habla](http://ipa-reader.xyz/)

# + [markdown] id="19eee353-6fd4-474a-86ca-8382ad51bf0f"
# ### Dataset: [IPA-dict](https://github.com/open-dict-data/ipa-dict) de open-dict

# + [markdown] id="18f45a54-5f64-408e-98f3-fc31114dc84a"
# - Diccionario de palabras para varios idiomas con su representación fonética
# - Representación simple, una palabra por renglon con el formato:
#
# ```
# [PALABRA][TAB][IPA]
#
# Ejemplos
# mariguana	/maɾiɣwana/
# zyuganov's   /ˈzjuɡɑnɑvz/, /ˈzuɡɑnɑvz/
# ```

# + [markdown] id="7cb52e47-d493-4b30-a991-ba5c4458d047"
# - [ISO language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
# - URL: `https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/<iso-lang>`

# + [markdown] id="4152e020-fbc0-4ec5-8d51-ccd2e8a089fc"
# #### Explorando el corpus 🗺️

# + executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1738875799202, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="25b595d7-7201-42bd-abb3-3acf9731d219"
IPA_URL = "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{lang}.txt"

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 234, "status": "ok", "timestamp": 1738875799645, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="3f45ba75-bbd3-4f13-8abf-b822fbf90dda" outputId="a60fa8cf-3907-4d33-ab25-c6fe3510675c"
import requests as r
from pprint import pprint as pp

response = r.get(IPA_URL.format(lang="en_US"))
response.text[:100]

ipa_data = response.text.split("\n")
#print(ipa_data[-4:])
ipa_data[-1]
pp(ipa_data[400:410])

# Puede haber mas de una transcipcion asociada a una palabra
print(ipa_data[-3].split("\t"))

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1738776291103, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="df50823b-11f6-4de3-b12c-8747cfa084bb" outputId="3a225a69-598a-4b30-fd7a-7d2c7b658887"
for data in ipa_data[300:500]:
    word, ipa = data.split('\t')
    representations = ipa.split(", ")
    if len(representations) >= 2:
        print(f"{word} --> {representations}")

# + [markdown] id="c671dbe4-1f99-443a-afb9-3f92951bef35"
# #### Obtención y manipulación

# + executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1738875803042, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="1fdc23af-9a0b-470d-a5f1-e5bddfa0b53e"
import http

def download_ipa_corpus(iso_lang: str) -> str:
    """Get ipa-dict file from Github

    Parameters:
    -----------
    iso_lang:
        Language as iso code

    Results:
    --------
    dict:
        Dictionary with words as keys and phonetic representation
        as values for a given lang code
    """
    print(f"Downloading {iso_lang}", end="::")
    response = r.get(IPA_URL.format(lang=iso_lang))
    status_code = response.status_code
    print(f"status={status_code}")
    if status_code != http.HTTPStatus.OK:
        print(f"ERROR on {iso_lang} :(")
        return ""
    return response.text


# + colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"elapsed": 165, "status": "ok", "timestamp": 1738776489203, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="4a2cab84-505b-4cdb-91a9-3e9c3410e1a0" outputId="38b3aab5-3312-47fe-faea-e0a32242dc82"
download_ipa_corpus("en_US").rstrip()[:50]


# + executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1738875806377, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="0a83a2a2-8e0e-4881-98f3-b9251a6be778"
def parse_response(response: str) -> dict:
    """Parse text response from ipa-dict to python dict

    Each row have the format:
    [WORD][TAB]/[IPA]/(, /[IPA]/)?

    Parameters
    ----------
    response: str
        ipa-dict raw text

    Returns
    -------
    dict:
        A dictionary with the word as key and the phonetic
        representations as value
    """
    ipa_list = response.rstrip().split("\n")
    result = {}
    for item in ipa_list:
        if item == '':
            continue
        item_list = item.split("\t")
        result[item_list[0]] = item_list[1]
    return result


# + colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"elapsed": 334, "status": "ok", "timestamp": 1738776559218, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="90b71697-bdec-45a8-8015-b8effc8b9f46" outputId="1ae24670-6d50-4752-d8cf-d25f5b0eb46d"
parse_response(download_ipa_corpus("en_US"))["ababa"]

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1067, "status": "ok", "timestamp": 1738776625486, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="dc377ece-bb15-488a-802b-e74a748e1da0" outputId="d517d62f-4388-4f4e-8717-3512f51e1e72"
es_mx_ipa = parse_response(download_ipa_corpus("es_MX"))


# + executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1738875808597, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="b834aaba-0716-41b5-935b-7f4a61e9da03"
def get_ipa_transcriptions(word: str, dataset: dict) -> list[str]:
    """Search for a word in an IPA phonetics dict

    Given a word this function return the IPA transcriptions

    Parameters:
    -----------
    word: str
        A word to search in the dataset
    dataset: dict
        A dataset for a given language code

    Returns
    -------
    list[str]:
        List with posible transcriptions if any,
        else an empty list
    """
    return dataset.get(word.lower(), "").split(", ")


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1738776696912, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="d89e45e2-5010-4701-84fd-e62b910233e7" outputId="0839146b-ed76-4c0a-c7cf-078ac7278791"
get_ipa_transcriptions("mayonesa", es_mx_ipa)

# + [markdown] id="37f69f04-ad55-4ca2-8bb1-d52fa58c4051"
# #### Obtengamos datasets

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7404, "status": "ok", "timestamp": 1738875818334, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="0828cb63-19b9-4cac-8df2-7dc8282fc4c3" outputId="6d41a0c2-f6fd-4546-94a2-316ddcf01eeb"
# Get datasets
dataset_es_mx = parse_response(download_ipa_corpus("es_MX"))
dataset_ja = parse_response(download_ipa_corpus("ja"))
dataset_en_us = parse_response(download_ipa_corpus("en_US"))
dataset_fr = parse_response(download_ipa_corpus("fr_FR"))

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1738776722896, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="694ef1c1-871e-407e-9ce5-5e177804f72f" outputId="f221b632-e77c-4649-e194-dbb49e2a644b"
# Simple query
get_ipa_transcriptions("beautiful", dataset_en_us)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 70, "status": "ok", "timestamp": 1738776751139, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="04a5fdbb-3acc-4e34-9685-61a591f2b598" outputId="13068823-f467-4526-d9d9-32b3f5bcf340"
# Examples
print(f"dog -> {get_ipa_transcriptions('dog', dataset_en_us)} 🐶")
print(f"mariguana -> {get_ipa_transcriptions('mariguana', dataset_es_mx)} 🪴")
print(f"猫 - > {get_ipa_transcriptions('猫', dataset_ja)} 🐈")
print(f"croissant -> {get_ipa_transcriptions('croissant', dataset_fr)} 🥐")

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 49, "status": "ok", "timestamp": 1738776807835, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="9001ea35-855f-499b-a5b9-3c70a0ba7397" outputId="91960028-7806-4544-99ba-af5ea517c68e"
# Diferentes formas de pronunciar
print(f"[es_MX] hotel | {dataset_es_mx['hotel']}")
print(f"[en_US] hotel | {dataset_en_us['hotel']}")

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 48, "status": "ok", "timestamp": 1738776863277, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="9fc9c19c-f6f4-4e35-9155-71aacaef1a05" outputId="16852ef9-832b-42ca-9390-717c42104153"
print(f"[ja] ホテル | {dataset_ja['ホテル']}")
print(f"[fr] hôtel | {dataset_fr['hôtel']}")

# + [markdown] id="41ca5bf8-93b1-4b10-9596-02bce9caccb8" jp-MarkdownHeadingCollapsed=true
# #### 🧙🏼‍♂️ Ejercicio: Obtener la distribución de frecuencias de los símbolos fonológicos para español

# + executionInfo={"elapsed": 1056, "status": "ok", "timestamp": 1738875787239, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="d616b9ef-396a-479b-adfd-1e15eab3fe37"
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


# + executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1738875787835, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="d3868dcd-eb1b-4e4a-9f52-ad7e2be72971"
def get_phone_symbols_freq(dataset: dict):
    freqs = defaultdict(int)
    ipas = [_.strip("/") for _ in dataset.values()]
    unique_ipas = set(ipas)
    for ipa in unique_ipas:
        for char in ipa:
            freqs[char] += 1
    return freqs


# + executionInfo={"elapsed": 1047, "status": "ok", "timestamp": 1738875819383, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="ijDBSOM5UB5i"
freqs_es = get_phone_symbols_freq(dataset_es_mx)
# Sorted by freq number (d[1]) descendent (reverse=True)
distribution_es = dict(sorted(freqs_es.items(), key=lambda d: d[1], reverse=True))
df_es = pd.DataFrame.from_dict(distribution_es, orient='index')

# + colab={"base_uri": "https://localhost:8080/", "height": 504} executionInfo={"elapsed": 677, "status": "ok", "timestamp": 1738875820064, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="YaooI8LQErwU" outputId="e351c900-21bd-4139-b458-73588039ea78"
df_es.plot(kind='bar', title="Distribución de caracteres foneticos para Español (MX)", xlabel="Caracter", ylabel="Frecuencia")

# + [markdown] id="aeb6c269-ad75-4e49-9090-247dc9a60231"
# #### Encontrar homófonos (palabras con el mismo sonido pero distina ortografía)
#
# - Ejemplos: Casa-Caza, Vaya-Valla

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 348, "status": "ok", "timestamp": 1738779300995, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="-UXEnSv6700t" outputId="0e02caa7-93da-4e37-c304-f1b96073f44d"
from collections import Counter

transcription_counts = Counter(dataset_es_mx.values())
duplicated_transcriptions = [transcription for transcription, freq in transcription_counts.items() if freq > 1]

for ipa in duplicated_transcriptions[-10:]:
    words = [word for word, transcription in dataset_es_mx.items() if transcription == ipa]
    print(f"{ipa} => {words}")

# + [markdown] id="a6e06a95-ceb6-49c0-bcbb-ff456976e510"
# #### Obteniendo todos los datos

# + id="9d92e8bd-53c9-4f2a-b926-b2de8ac19357"
lang_codes = {
    "ar": "Arabic (Modern Standard)",
    "de": "German",
    "en_UK": "English (Received Pronunciation)",
    "en_US": "English (General American)",
    "eo": "Esperanto",
    "es_ES": "Spanish (Spain)",
    "es_MX": "Spanish (Mexico)",
    "fa": "Persian",
    "fi": "Finnish",
    "fr_FR": "French (France)",
    "fr_QC": "French (Québec)",
    "is": "Icelandic",
    "ja": "Japanese",
    "jam": "Jamaican Creole",
    "km": "Khmer",
    "ko": "Korean",
    "ma": "Malay (Malaysian and Indonesian)",
    "nb": "Norwegian Bokmål",
    "nl": "Dutch",
    "or": "Odia",
    "ro": "Romanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tts": "Isan",
    "vi_C": "Vietnamese (Central)",
    "vi_N": "Vietnamese (Northern)",
    "vi_S": "Vietnamese (Southern)",
    "yue": "Cantonese",
    "zh_hans": "Mandarin (Simplified)",
    "zh_hant": "Mandarin (Traditional)"
}
iso_lang_codes = list(lang_codes.keys())


# + id="aaf29cd0-be3c-4821-a608-71275da4852e"
def get_corpora() -> dict:
    """Download corpora from ipa-dict github

    Given a list of iso lang codes download available datasets.

    Returns
    -------
    dict
        Lang codes as keys and dictionary with words-transcriptions
        as values
    """
    return {
        code: parse_response(download_ipa_corpus(code))
         for code in iso_lang_codes
        }


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 33883, "status": "ok", "timestamp": 1738779595792, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="f04bc065-584b-4373-bc7f-9e328793bbe4" outputId="9195f012-6ba8-4c12-b0ed-1ed2f419a266"
data = get_corpora()

# + [markdown] id="41cc36cd-7919-4372-ada2-d9f52432b56c"
# #### Sistema de búsqueda (naïve)

# + id="26b1bcf5-f8ca-42ab-995a-e28d9fcc47d3"
from rich import print as rprint
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text


# + id="24424a3c-70df-416f-8923-6bcd2a435007"
def get_formated_string(code: str, name: str):
    return f"[b]{name}[/b]\n[yellow]{code}"


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 108841, "status": "ok", "timestamp": 1738779804332, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="bd09c65f-c559-4571-98a8-20c0d2c0308e" outputId="14a4e15e-43b2-4f83-8081-f5562967c8d1"
rprint(Panel(Text("Representación fonética de palabras", style="bold", justify="center")))
rendable_langs = [Panel(get_formated_string(code, lang), expand=True) for code, lang in lang_codes.items()]
rprint(Columns(rendable_langs))

lang = input("lang>> ")
rprint(f"Selected language: {lang_codes[lang]}") if lang else rprint("Adios 👋🏼")
while lang:
    sub_dataset = data[lang]
    query = input(f"  [{lang}]word>> ")
    results = get_ipa_transcriptions(query, sub_dataset)
    rprint(query, " | ", ", ".join(results))
    while query:
        query = input(f"  [{lang}]word>> ")
        if query:
            results = get_ipa_transcriptions(query, sub_dataset)
            rprint(query, " | ", ", ".join(results))
    lang = input("lang>> ")
    rprint(f"Selected language: {lang_codes[lang]}") if lang else rprint("Adios 👋🏼")

# + [markdown] id="dc8b18ff-9d70-49a6-98c1-7feb7d0c268a"
# #### 👩‍🔬 Ejercicio: Obtener palabras con pronunciación similar

# + id="a4116005-8fb4-454d-a2ab-e2f083eda000"
from collections import defaultdict

def get_rhyming_patterns(sentence: str, dataset: dict) -> dict[str, list]:
    words = sentence.split()
    word_ipa_map = {}
    for word in words:
        ipa_transcriptions = get_ipa_transcriptions(word, dataset)
        # Remove "/" char from transcriptions
        word_ipa_map.update({word: [_.strip("/") for _ in ipa_transcriptions]})

    rhyming_patterns = defaultdict(list)
    for word, ipas in word_ipa_map.items():
        for ipa in ipas:
            # Getting last 2 elements of the ipa representation
            pattern = ipa[-2:]
            rhyming_patterns[pattern].append(word)
    return rhyming_patterns

def display_rhyming_patterns(patterns: dict[str, list]) -> None:
    for pattern, words in patterns.items():
        if len(set(words)) > 1:
            print(f"{pattern}:: {', '.join(words)}")


# + [markdown] id="f4daacb3-544b-4b3a-ab77-23b2fc4dd07e"
# #### Testing

# + [markdown] id="3HoQEf8i8qTo"
# ```
# ɣo:: juego, fuego
# on:: con, corazón
# ʎa:: brilla, orilla
# ```

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 71, "status": "ok", "timestamp": 1738779950829, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="057bb91d-5bef-47b4-ba82-42559f457c2b" outputId="0435b8cf-4395-438b-a78f-0f872b9cb287"
#sentence = "There once was a cat that ate a rat and after that sat on a yellow mat"
#sentence = "the cat sat on the mat and looked at the rat."
sentence = "If you drop the ball it will fall on the doll"
#sentence = "cuando juego con fuego siento como brilla la orilla de mi corazón"

dataset = data.get("en_US")
rhyming_words = get_rhyming_patterns(sentence, dataset)
display_rhyming_patterns(rhyming_words)

# + [markdown] id="86f07e0f-bfe4-4a14-be9d-a7b0b7661260"
# #### Material extra (fonética)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13053, "status": "ok", "timestamp": 1738780054335, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="64dc5e71-449b-4fdf-a3f4-d0f1776c5bbf" outputId="8b70a454-65b1-447d-aee3-73b1933b134b"
# ! apt-get install espeak -y

# + id="c87a1b9d-848c-488f-b2e4-1782f07bc557"
# !espeak -v es "Hola que hace" -w mi-prueba.wav

# + [markdown] id="9fc31a40-1d6e-4c56-b07e-74a0c47a89c4" jp-MarkdownHeadingCollapsed=true
# ## Morfología

# + [markdown] id="GJ10fzsXvFSS"
# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Flexi%C3%B3nGato-svg.svg/800px-Flexi%C3%B3nGato-svg.svg.png" height=300></center>
#
# > De <a href="//commons.wikimedia.org/wiki/User:KES47" class="mw-redirect" title="User:KES47">KES47</a> - <a href="//commons.wikimedia.org/wiki/File:Flexi%C3%B3nGato.png" title="File:FlexiónGato.png">File:FlexiónGato.png</a> y <a href="//commons.wikimedia.org/wiki/File:Nuvola_apps_package_toys_svg.svg" title="File:Nuvola apps package toys svg.svg">File:Nuvola apps package toys svg.svg</a>, <a href="http://www.gnu.org/licenses/lgpl.html" title="GNU Lesser General Public License">LGPL</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=27305101">Enlace</a>

# + [markdown] id="9677e8f1-aa3e-4f9c-8c7c-1422ea9ca913"
# El análisis morfológico es la determinación de las partes que componen la palabra y su representación lingüística, es una especie de etiquetado
#
# Los elementos morfológicos son analizados para:
#
# - Determinar la función morfológica de las palabras
# - Hacer filtrado y pre-procesamiento de text

# + [markdown] id="d962a952-6fa8-4410-82b6-8e7c6ba2f7e4"
# ### Análisis morfológico basado en reglas

# + [markdown] id="cf70678d-f1d1-401e-aa23-da90a2ea7eaa"
# Recordemos que podemos hacer un analizador morfológico haciendo uso de un transductor que vaya leyendo y haciendo transformaciones en una cadena. Formalmente:
#
# * $Q = \{q_0, \ldots, q_T\}$ conjunto finito de estados.
# * $\Sigma$ es un alfabeto de entrada.
# * $q_0 \in Q$ es el estado inicial.
# * $F \subseteq Q$ es el conjunto de estados finales.
#
# Un transductor es una 6-tupla $T = (Q, \Sigma, \Delta, q_0, F, \sigma)$ tal que
#
# * $\Delta$ es un alfabeto de salida teminal
# * $\Sigma$ es un alfabeto de entrada no terminal
# * $\sigma: Q \times \Sigma \times \Delta \longrightarrow Q$ función de transducción

# + [markdown] id="cdf136a8-6a9d-4434-b533-191553db242b"
# #### EJEMPLO: Parsing con expresiones regulares

# + [markdown] id="605d64c5-e102-4972-b5f6-92e0c08b495b"
# Con fines de prácticidad vamos a _imitar_ el comportamiento de un transductor utilizando el modulo de python `re`

# + [markdown] id="cb409f23-1456-40bd-8fdd-997a862ad190"
# La estructura del sustantivo en español es:
#
# ` BASE+AFIJOS (marcas flexivas)   --> Base+DIM+GEN+NUM`

# + id="1e2959df-96a4-40f5-a932-108abad269be"
palabras = [
    'niño',
    'niños',
    'niñas',
    'niñitos',
    'gato',
    'gatos',
    'gatitos',
    'perritos',
    'paloma',
    'palomita',
    'palomas',
    'flores',
    'flor',
    'florecita',
    'lápiz',
    'lápices',
    # 'chiquitititititos',
    #'curriculum', # curricula
    #'campus', # campi
]

# + id="66599e2e-d67b-49e2-9f80-0a20e756ca19"
import re

def morph_parser_rules(words: list[str]) -> list[str]:
    """Aplica reglas morfológicas a una lista de palabras para realizar
    un análisis morfológico.

    Parameters:
    ----------
    words : list of str
        Lista de palabras a las que se les aplicarán las reglas morfológicas.

    Returns:
    -------
    list of str
        Una lista de palabras después de aplicar las reglas morfológicas.
    """

    # Lista para guardar las palabras parseadas
    morph_parsing = []

    # Reglas que capturan ciertos morfemas
    # {ecita, itos, as, os}
    for w in words:
        # ecit -> DIM
        R0 = re.sub(r'([^ ]+)ecit([a|o|as|os])', r'\1-DIM\2', w)
        # it -> DIM
        R1 = re.sub(r'([^ ]+)it([a|o|as|os])', r'\1-DIM\2', R0)
        # a(s) -> FEM
        R2 = re.sub(r'([^ ]+)a(s)', r'\1-FEM\2', R1)
        # a -> FEM
        R3 = re.sub(r'([^ ]+)a\b', r'\1-FEM', R2)
        # o(s) -> MSC
        R4 = re.sub(r'([^ ]+)o(s)', r'\1-MSC\2', R3)
        # o .> MSC
        R5 = re.sub(r'([^ ]+)o\b', r'\1-MSC', R4)
        # es -> PL
        R6 = re.sub(r'([^ ]+)es\b', r'\1-PL', R5)
        # s -> PL
        R7 = re.sub(r'([^ ]+)s\b', r'\1-PL', R6)
        # Sustituye la c por z cuando es necesario
        parse = re.sub(r'c-', r'z-', R7)

        # Guarda los parseos
        morph_parsing.append(parse)
    return morph_parsing


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1738780567898, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="85c9648c-0c09-4d19-b52b-f020943caf5a" outputId="6804fc5f-edb7-423c-9bf3-96ec7fccc605"
morph_parsing = morph_parser_rules(palabras)
for palabra, parseo in zip(palabras, morph_parsing):
    print(palabra, "-->", parseo)

# + [markdown] id="3a104dae-e815-4271-9ff2-96802d50df9e"
# #### Preguntas 🤔
# - ¿Qué pasa con las reglas en lenguas donde son más comunes los prefijos y no los sufijos?
# - ¿Cómo podríamos identificar características de las lenguas?

# + [markdown] id="8b3d6dac-6b02-45db-b352-a549a25fdabe"
# #### Herramientas para hacer sistemas de análisis morfológico basados en reglas

# + [markdown] id="2ff2d85d-64c1-4272-9d6d-cf3599094588"
# - [Apertium](https://en.wikipedia.org/wiki/Apertium)
# - [Foma](https://github.com/mhulden/foma/tree/master)
# - [Helsinki Finite-State Technology](https://hfst.github.io/)
# - Ejemplo [proyecto](https://github.com/apertium/apertium-yua) de analizador morfológico de Maya Yucateco
# - Ejemplo normalizador ortográfico del [Náhuatl](https://github.com/ElotlMX/py-elotl/tree/master)
#
#
# También se pueden utilizar diferentes métodos de aprendizaje de máquina para realizar análisis/generación morfológica. En los últimos años ha habido un shared task de [morphological reinflection](https://github.com/sigmorphon/2023InflectionST) para poner a competir diferentes métodos

# + [markdown] id="d5b878ce-f60a-4069-b618-fcb0d4e77256"
# ### Segmentación morfológica

# + [markdown] id="bcdc6126-3d69-4f90-9cfe-9dd6845525d5"
# #### Corpus: [SIGMORPHON 2022 Shared Task on Morpheme Segmentation](https://github.com/sigmorphon/2022SegmentationST/tree/main)

# + [markdown] id="78102f5e-2bcc-41cd-981f-514d786cb9be"
# - Shared task donde se buscaba convertir las palabras en una secuencia de morfemas
# - Dividido en dos partes:
#     - Segmentación a nivel de palabras (nos enfocaremos en esta)
#     - Segmentación a nivel oraciones

# + [markdown] id="c6b0c352-e6ac-49db-9b3a-74da7db6ddad"
# #### Track: words
#
# | word class | Description                      | English example (input ==> output)     |
# |------------|----------------------------------|----------------------------------------|
# | 100        | Inflection only                  | played ==> play @@ed                   |
# | 010        | Derivation only                  | player ==> play @@er                   |
# | 101        | Inflection and Compound          | wheelbands ==> wheel @@band @@s        |
# | 000        | Root words                       | progress ==> progress                  |
# | 011        | Derivation and Compound          | tankbuster ==> tank @@bust @@er        |
# | 110        | Inflection and Derivation        | urbanizes ==> urban @@ize @@s          |
# | 001        | Compound only                    | hotpot ==> hot @@pot                   |
# | 111        | Inflection, Derivation, Compound | trackworkers ==> track @@work @@er @@s

# + [markdown] id="84e2235d-f919-4c5d-832e-93551ba9ac4b"
# #### Explorando el corpus

# + colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 1103, "status": "ok", "timestamp": 1738780988950, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="1a59cbf7-de9d-4618-8229-5cd369fa1c07" outputId="740f54b5-52b8-4665-863a-0969ec99fd86"
response = r.get("https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/spa.word.test.gold.tsv")
response.text[:100]

# + colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 45, "status": "ok", "timestamp": 1738780996487, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="17faa0d5-2d5f-4035-bf65-c0e5b256aa0f" outputId="29c6287e-391d-4346-f843-874766bb1a9c"
raw_data = response.text.split("\n")
raw_data[-2]

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1738780999647, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="a1814873-5c65-4f34-9c59-01ccdf130089" outputId="80ca7d49-9053-4b41-a383-e1fc40749cb5"
element = raw_data[2].split("\t")
element

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1738781001712, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="99c1dfb3-a1ad-4a78-b26b-8a560894f058" outputId="da0bb22f-5ea6-4e65-e8fe-a8acfd829fc2"
element[1].split()

# + id="bb87e719-41c2-4e22-b9bf-5303902be165"
LANGS = {
    "ces": "Czech",
    "eng": "English",
    "fra": "French",
    "hun": "Hungarian",
    "spa": "Spanish",
    "ita": "Italian",
    "lat": "Latin",
    "rus": "Russian",
}
CATEGORIES = {
    "100": "Inflection",
    "010": "Derivation",
    "101": "Inflection, Compound",
    "000": "Root",
    "011": "Derivation, Compound",
    "110": "Inflection, Derivation",
    "001": "Compound",
    "111": "Inflection, Derivation, Compound"
}


# + id="721ea2fa-24d8-4035-b0a5-87c86f821c6d"
def get_track_files(lang: str, track: str = "word") -> list[str]:
    """Genera una lista de nombres de archivo del shared task

    Con base en el idioma y el track obtiene el nombre de los archivos
    para con información reelevante para hacer análisis estadístico.
    Esto es archivos .test y .dev

    Parameters:
    ----------
    lang : str
        Idioma para el cual se generarán los nombres de archivo.
    track : str, optional
        Track del shared task de donde vienen los datos (por defecto es "word").

    Returns:
    -------
    list[str]
        Una lista de nombres de archivo generados para el idioma y la pista especificados.
    """
    return [
        f"{lang}.{track}.test.gold",
        f"{lang}.{track}.dev",
    ]


# + id="f583e168-1f5d-4426-9789-5fac8b2b221c"
def get_raw_corpus(files: list) -> list:
    """Descarga y concatena los datos de los archivos tsv desde una URL base.

    Parameters:
    ----------
    files : list
        Lista de nombres de archivos (sin extensión) que se descargarán
        y concatenarán.

    Returns:
    -------
    list
        Una lista que contiene los contenidos descargados y concatenados
        de los archivos tsv.
    """
    result = []
    for file in files:
        print(f"Downloading {file}.tsv", end=" ")
        response = r.get(f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{file}.tsv")
        print(f"status={response.status_code}")
        lines = response.text.split("\n")
        result.extend(lines[:-1])
    return result


# + id="54de3b3d-be08-437d-b4ee-ff55d4fee2a9"
import pandas as pd

def raw_corpus_to_dataframe(corpus_list: list, lang: str) -> pd.DataFrame:
    """Convierte una lista de datos de corpus en un DataFrame

    Parameters:
    ----------
    corpus_list : list
        Lista de líneas del corpus a convertir en DataFrame.
    lang : str
        Idioma al que pertenecen los datos del corpus.

    Returns:
    -------
    pd.DataFrame
        Un DataFrame de pandas que contiene los datos del corpus procesados.
    """
    data_list = []
    for line in corpus_list:
        try:
            word, tagged_data, category = line.split("\t")
        except ValueError:
            # Caso donde no existe la categoria
            word, tagged_data = line.split("\t")
            category = "NOT_FOUND"
        morphemes = tagged_data.split()
        data_list.append({"words": word, "morph": morphemes, "category": category, "lang": lang})
    df = pd.DataFrame(data_list)
    df["word_len"] = df["words"].apply(lambda x: len(x))
    df["morph_count"] = df["morph"].apply(lambda x: len(x))
    return df


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4913, "status": "ok", "timestamp": 1738781137592, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="fe645a77-f8ca-4bf1-b11e-2274b78ba6d1" outputId="d7e50464-31a2-4979-ca32-1ecdb7c07e5c"
files = get_track_files("spa")
raw_spa = get_raw_corpus(files)
df = raw_corpus_to_dataframe(raw_spa, lang="spa")

# + colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 52, "status": "ok", "timestamp": 1738781092482, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="774a3b39-3a37-43d5-afb4-5d98154afe9e" outputId="c99841e0-ea76-4343-9cca-0390ee68910c"
df.head()

# + [markdown] id="0ffef737-bd71-43ed-be6d-a357819ab7c8"
# #### Análisis cuantitativo para el Español

# + colab={"base_uri": "https://localhost:8080/", "height": 384} executionInfo={"elapsed": 152, "status": "ok", "timestamp": 1738781120781, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="e02c54b1-8b19-4a37-8a64-37749beb0418" outputId="979c36e3-eace-4040-f56a-c749ad5e0fb3"
print("Total unique words:", len(df["words"].unique()))
df["category"].value_counts().head(30)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 44, "status": "ok", "timestamp": 1738781169018, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="39965a92-4719-4043-919b-3b99dca0b8f9" outputId="bac4b6a4-efa9-40ce-ac86-51939ec0d6bd"
df["word_len"].mean()

# + colab={"base_uri": "https://localhost:8080/", "height": 472} executionInfo={"elapsed": 235, "status": "ok", "timestamp": 1738781188310, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="baf02d64-9bc1-4135-9f05-e523728ba269" outputId="d9774960-1726-44e7-cb31-d7061716f52f"
from matplotlib import pyplot as plt

plt.hist(df['word_len'], bins=10, edgecolor='black')
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.title('Word Length Distribution')
plt.show()


# + id="c902ecf4-889a-4082-b4b8-0cb89ba9b16c"
def plot_histogram(df, kind, lang):
    """Genera un histograma de frecuencia para una columna específica
    en un DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos para generar el histograma.
    kind : str
        Nombre de la columna para la cual se generará el histograma.
    lang : str
        Idioma asociado a los datos.

    Returns:
    -------
    None
        Esta función muestra el histograma usando matplotlib.
    """
    counts = df[kind].value_counts().head(30)
    plt.bar(counts.index, counts.values)
    plt.xlabel(kind)
    plt.ylabel('Frequency')
    plt.title(f'{kind} Frequency Graph for {lang}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"elapsed": 291, "status": "ok", "timestamp": 1738781219706, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="1f36d2ce-cca3-49f0-b989-69dae98f9646" outputId="b3648c42-99eb-452c-fb85-d73218829014"
plot_histogram(df, "category", "spa")

# + [markdown] id="408683e9-5d9b-44a6-bcb5-d2ad29bf7903"
# #### Morfosintaxis

# + [markdown] id="10233e8c-b359-4ad7-ab6f-8d6d0c600a39"
# - Etiquetas que hacen explícita la funcion gramatical de las palabras en una oración
# - Determina la función de la palabra dentro la oración (por ello se le llama Partes del Discurso)
# - Se le conoce tambien como **Análisis morfosintáctico**: es el puente entre la estructura de las palabras y la sintaxis
# - Permiten el desarrollo de herramientas de NLP más avanzadas
# - El etiquetado es una tarea que se puede abordar con técnicas secuenciales, por ejemplo, HMMs, CRFs, Redes neuronales

# + [markdown] id="8db57418-b7bc-4dc1-972d-aef484e9ea48"
# <center><img src="https://byteiota.com/wp-content/uploads/2021/01/POS-Tagging.jpg" height=500 width=500></center

# + [markdown] id="628dd2cd-c0b4-4b12-aa08-b74e5a81579c"
# #### Ejemplo
#
# > El gato negro rie malvadamente
#
# - El - DET
# - gato - NOUN
# - negro - ADJ
# - ríe - VER
#
# <center><img src="https://i.pinimg.com/originals/0e/f1/30/0ef130b255ea704625b2ad473701dee5.gif"></center

# + [markdown] id="522e2222-af00-4315-82ce-11534116f0b8"
# ### Etiquetado POS usando Conditional Random Fields (CRFs)

# + [markdown] id="2276eafd-a3e4-48a8-b97a-359503a7d66f"
# - Modelo de gráficas **no dirigido**. Generaliza los *HMM*
#     - Adiós a la *Markov assuption*
#     - Podemos tener cualquier dependencia que queramos entre nodos
#     - Nos enfocaremos en un tipo en concreto: *LinearChain-CRFs* ¡¿Por?!
#
# <center><img width=300 src="https://i.kym-cdn.com/entries/icons/original/000/032/676/Unlimited_Power_Banner.jpg"></center>
#

# + [markdown] id="c5a1bff5-1f06-416b-9244-c4eab4dd989a"
# - Modela la probabilidad **condicional** $P(Y|X)$
#     - Modelo discriminativo
#     - Probabilidad de un estado oculto dada **toda** la secuecia de entrada
# ![homer](https://media.tenor.com/ul0qAKNUm2kAAAAd/hiding-meme.gif)

# + [markdown] id="74beab61-39ec-43bf-8ca8-44cbb9d62149"
# - Captura mayor **número de dependencias** entre las palabras y captura más características
#     - Estas se definen en las *feature functions* 🙀
# - El entrenamiento se realiza aplicando gradiente decendente y optimización con algoritmos como [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
#
#
# <center><img src="https://iameo.github.io/images/gradient-descent-400.gif"></center>
#

# + [markdown] id="4653baf1-edc2-4813-be95-643a1b0f60f7"
# $P(\overrightarrow{y}|\overrightarrow{x}) = \frac{1}{Z} \displaystyle\prod_{i=1}^N exp\{w^T ⋅ \phi(y_{i-1}, y_i, \overrightarrow{x}, i)\}$
#
# Donde:
# - $\overrightarrow{y}$ = Etiquetas POS
# - $\overrightarrow{x}$ = Palabras en una oración
# - $w^T$ = Vector de pesos a aprender
# - $\phi$ = Vector de *Features*
#     - Calculado con base en un conjunto de *feature functions*
# - $i$ = la posición actual en la oración
# - $Z$ = factor de normalización

# + [markdown] id="5c13a366-4bc3-425c-8db4-bb2986dc2e8f"
# ![](https://aman.ai/primers/ai/assets/conditional-random-fields/Conditional_Random_Fields.png)
#
# Tomado de http://www.davidsbatista.net/blog/2017/11/13/Conditional_Random_Fields/

# + [markdown] id="5ad8c1dc-4c5b-41f8-95c2-40be89f8f07f"
# #### Feature functions
#
# $\phi(y_{i-1}, y_i, \overrightarrow{x}, i)$
#
# - Parte fundamental de los CRFs
# - Cuatro argumentos:
#     - Todos los datos observables $\overrightarrow{x}$ (conectar $x$ con cualquier $y$)
#     - El estado oculto anterior $y_{i-1}$
#     - El estado oculto actual $y_i$
#     - El index del timestamp $i$
#         - Cada feature list puede tener diferentes formas

# + [markdown] id="c1cfd87f-d4d2-4501-bd47-5a7dc843db2b"
# - Aqui es donde esta la flexibilidad del modelo
# - Tantas features como querramos, las que consideremos que pueden ayudar a que el modelo tenga un mejor desempeño
#     - Intimamente ligadas a la lengua. Para mejor desempeño se debe hacer un análisis de sus características.
# - Ejemplo:
#
# ```python
# [
#     "word.lower()",
#     "EOS",
#     "BOS",
#     "postag",
#     "pre-word",
#     "nxt-word",
#     "word-position",
#     ...
# ]
# ```

# + [markdown] id="1c50c45f-9e37-42e0-9cfe-1a78429660ed"
# ### Implementación de CRFs

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9150, "status": "ok", "timestamp": 1738781649643, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="ca05605b-7776-4fc8-9e6a-7541a28d659f" outputId="bf87417f-828e-499d-98e6-50f438c9dc9a"
# !pip install nltk
# !pip install scikit-learn
# !pip install -U sklearn-crfsuite

# + [markdown] id="b2e0fb29-beee-4a0a-8010-0f62db206981"
# #### Obteniendo otro corpus más

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3386, "status": "ok", "timestamp": 1738781673696, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="5651f7bc-21de-4379-93b8-07499f6df74a" outputId="86fd3488-9802-4fca-b9fd-241091a3b405"
import nltk

# Descargando el corpus cess_esp: https://www.nltk.org/book/ch02.html#tab-corpora
nltk.download('cess_esp')

# + id="1ab197a8-2ea3-4d85-acf8-745d6f99be83"
from nltk.corpus import cess_esp
# Cargando oraciones
corpora = cess_esp.tagged_sents()

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1738781678109, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="We80idF3qUIb" outputId="57544344-2f95-44f4-e8b7-486c7b6631b3"
corpora[1]

# + id="45d7a052-6e9e-47ca-8fc2-13ff05963b53"
import requests

def get_tags_map() -> dict:
    tags_raw = requests.get("https://gist.githubusercontent.com/vitojph/39c52c709a9aff2d1d24588aba7f8155/raw/af2d83bc4c2a7e2e6dbb01bd0a10a23a3a21a551/universal_tagset-ES.map").text.split("\n")
    tags_map = {line.split("\t")[0].lower(): line.split("\t")[1] for line in tags_raw}
    return tags_map

def map_tag(tag: str, tags_map=get_tags_map()) -> str:
    return tags_map.get(tag.lower(), "N/F")

def parse_tags(corpora: list[list[tuple]]) -> list[list[tuple]]:
    result = []
    for sentence in corpora:
        print
        result.append([(word, map_tag(tag)) for word, tag in sentence if tag not in ["Fp", "Fc", "Fpa", "Fpt"]])
    return result


# + id="c07cc1fb-8b6d-4a3d-a637-93c15bcbbfc6"
corpora = parse_tags(corpora)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1738781724293, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="d2347079-1504-458b-9743-5bc4dfa7d1e6" outputId="b4a40eb1-b9ff-423c-914f-b315097e7ee6"
corpora[0]


# + [markdown] id="d7864024-2275-4cf6-93f4-16c6b0f54451"
# #### Feature lists

# + id="2b6ab1f8-71fc-4862-ac2a-35fcb3ca5b2a"
def word_to_features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'prefix_1': word[:1],
        'prefix_2': word[:2],
        'suffix_1': word[-1:],
        'suffix_2': word[-2:],
        'word_len': len(word)
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    return features

# Extract features and labels
def sent_to_features(sent) -> list:
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent) -> list:
    return [label for token, label in sent]


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1738781787425, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="59da4b3c-7d1d-4725-936c-0afdbf74137f" outputId="8c9c0967-e2d2-4681-849b-dc4a9796bfff"
# ¿Cuantas oraciones tenemos disponibles?
len(corpora)

# + id="7266b709-0287-4f48-a62f-12ca9d7aaffb"
# Preparando datos para el CRF
X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in corpora]
y = [[pos for _, pos in sent] for sent in corpora]

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1738781801148, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="f7f68f41-8ad8-42c4-b20d-626561047433" outputId="884befed-36dd-465f-ff62-ee8fee034187"
# Exploración de data estructurada
X[0]

# + id="e6bdf707-c7fc-408b-9d81-65528e068cbb"
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# + id="49503189-368e-4d83-abd2-b06433bae3e8"
assert len(X_train) + len(X_test) == len(corpora), "Something wrong with my split :("
assert len(y_train) + len(y_test) == len(corpora), "Something wrong with my split :("

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25901, "status": "ok", "timestamp": 1738781912893, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="cba343a2-482f-4c67-a842-704ab5fc6f3e" outputId="3355fa0f-d2f3-4c66-de0b-595c8b754495" jupyter={"outputs_hidden": true}
from inspect import Attribute
from sklearn_crfsuite import CRF
# Initialize and train the CRF tagger: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, verbose=True)
try:
    crf.fit(X_train, y_train)
except AttributeError as e:
    print(e)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 924, "status": "ok", "timestamp": 1738781918814, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="983f2a29-b455-4ca3-8115-eb4962e25481" outputId="d59aff27-781e-4b97-f262-7f39845c7e88"
from sklearn.metrics import classification_report
y_pred = crf.predict(X_test)

# Flatten the true and predicted labels
y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

# Evaluate the model
report = classification_report(y_true=y_test_flat, y_pred=y_pred_flat)
print(report)


# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Tarea 1: Niveles del lenguaje
#
# ### FECHA DE ENTREGA: 16 de Febrero 2025 at 11:59pm
#
# ### Fonética
#
# 1. Si tenemos un sistema de búsqueda que recibe una palabra ortográfica y devuelve sus transcripciones fonológicas, proponga una solución para los casos en que la palabra buscada no se encuentra en el lexicón/diccionario. *¿Cómo devolver o aproximar su transcripción fonológica?*
#   - Reutiliza el sistema de búsqueda visto en clase y mejoralo con esta funcionalidad

# +
def edit_distance(word1: str, word2: str) -> int:
    """
    Calcula la distancia de edición (distancia de Levenshtein) entre dos palabras.
    
    La distancia de edición es el número mínimo de operaciones requeridas para transformar
    una palabra en otra. Las operaciones permitidas son:
    - Inserción de un carácter
    - Eliminación de un carácter
    - Sustitución de un carácter
    
    Args:
        word1 (str): Primera palabra.
        word2 (str): Segunda palabra.
    
    Returns:
        int: La distancia de edición entre word1 y word2.
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # Inserciones
            elif j == 0:
                dp[i][j] = i  # Eliminaciones
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No hay cambio
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # Eliminación
                                   dp[i][j - 1],    # Inserción
                                   dp[i - 1][j - 1]) # Sustitución
    
    return dp[m][n]

def get_ipa_transcriptions_boosted(word: str, dataset: dict) -> list[str]:
    """Search for a word in an IPA phonetics dict

    Given a word this function return the IPA transcriptions or the transcriptions of the words with the lowest edit distance

    Parameters:
    -----------
    word: str
        A word to search in the dataset
    dataset: dict
        A dataset for a given language code

    Returns
    -------
    list[str]:
        List with posible transcriptions if any,
        else the transcriptions of the word with the lowest edit distance
    """
    transcript = dataset.get(word.lower(), "")
    if not transcript:
        nearest_five = [(w, edit_distance(word.lower(), w)) for w in dataset.keys()]
        print(nearest_five)
        nearest_five = sorted(nearest_five, key=lambda x: x[1])[:5]
        return [(w[0], dataset.get(w[0], "").split(", ")) for w in nearest_five], False
    return transcript.split(", "), True


# +
rprint(Panel(Text("Representación fonética de palabras", style="bold", justify="center")))
rendable_langs = [Panel(get_formated_string(code, lang), expand=True) for code, lang in lang_codes.items()]
rprint(Columns(rendable_langs))

lang = input("lang>> ")
rprint(f"Selected language: {lang_codes[lang]}") if lang else rprint("Adios 👋🏼")
while lang:
    sub_dataset = data[lang]
    query = input(f"  [{lang}]word>> ")
    results, exact_match = get_ipa_transcriptions_boosted(query, sub_dataset)
    if not exact_match:
        rprint("Quizás quisiste decir:")
        for result in results: rprint(result[0], " | ", ", ".join(result[1]))
    else:
        rprint(query, " | ", ", ".join(results))
    while query:
        query = input(f"  [{lang}]word>> ")
        if query:
            results, exact_match = get_ipa_transcriptions_boosted(query, sub_dataset)
            if not exact_match:
                rprint("Quizás quisiste decir:")
                for result in results: rprint(result[0], " | ", ", ".join(result[1]))
            else:
                rprint(query, " | ", ", ".join(results))
    lang = input("lang>> ")
    rprint(f"Selected language: {lang_codes[lang]}") if lang else rprint("Adios 👋🏼")

# + [markdown] id="1266180c-54ca-433c-bf77-e7052df67291"
# ### Morfología
#
# 2. Obtenga los datos de `test` y `dev` para todas las lenguas disponibles en el Shared Task SIGMORPHON 2022 y haga lo siguiente:
#     - En un plot de 4 columnas y 2 rows muestre las siguientes distribuciones (un subplot por lengua):
#         - Plot 1: distribución de longitud de palabras
#         - Plot 2: distribución de la cuenta de morfemas
#         - Plot 3: distribución de categorias (si existe para la lengua)
#     - Realice una función que imprima por cada lengua lo siguiente:
#         - Total de palabras
#         - La longitud de palabra promedio
#         - La cuenta de morfemas promedio
#         - La categoría más común
#     - Con base en esta información elabore una conclusión lingüística sobre la morfología de las lenguas analizadas.
# -



# ### EXTRA:
#
# - Imprimir la [matríz de confusión](https://en.wikipedia.org/wiki/Confusion_matrix) para el etiquetador CRFs visto en clase y elaborar una conclusión sobre los resultados

# + id="oW_JEkMlsSIs"

