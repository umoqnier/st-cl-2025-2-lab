#!/usr/bin/env python
# coding: utf-8

# # Alumno: Vázquez Martínez Fredin Alberto
# 
# # 2: Propiedades estadísticas de la lengua
# 
# ## Fecha de entrega: 2 de Marzo de 2025 11:59pm
# 
# 1. Verificar si la ley de Zipf se cumple en un lenguaje artificial creado por ustedes.
#   * Ejemplo: Un "lenguaje artificial" podría ser simplemente un texto donde las secuencias de caracteres fueron generadas aleatoriamente.
# 
# 
# ### **Desarrollo**
# Para generar el lenguaje la estrategía a usar es por medio de dos pasos:
# 
# 1. Crear Vocabulario
#   * Generar un número aleatorio para definir la longitud de la palabra, sería un número k aleatorio.
#   * Escoger k carácteres del código ASCII.
#   * Cada palabra generada se va guardar en un arreglo, este será nuestro vocabulario.
# 2. Generar texto
#   * Se espera que el usuario ingrese la longitud del texto a generar, siendo l.
#   * Se van a escoger l palabras de manera aleatoria de nuestro vocabulario creado anteriormente.
# 
# Se genera así un texto creado con nuestro lenguaje artificial creado de manera aleatoria.
# 

# In[ ]:


# Bibliotecas
import numpy as np
import pandas as pd
import string
import random
from collections import Counter
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
from collections import Counter
from prettytable import PrettyTable


# In[ ]:


def generador_lenguaje(longitud_texto, tamano_vocabulario):
  """
  Genera un texto artificial utilizando un vocabulario aleatorio.

  La función crea un vocabulario de palabras aleatorias y luego genera un texto
  de la longitud especificada seleccionando palabras al azar del vocabulario.

  Parameters
  ----------
  longitud_texto : int
      Longitud del texto artificial a generar.
  tamano_vocabulario : int
      Tamaño del vocabulario de palabras aleatorias a crear.

  Returns
  -------
  list
      Lista de palabras que conforman el texto generado.
  """
  np.random.seed(42)
  texto = []
  vocabulario = []

  for i in range(tamano_vocabulario):
    palabra = ""
    k = random.randint(2,10) # Vamos que se puede generar hasta una palabra de 10 caracteres
    for i in range(k):
      palabra += random.choice(string.ascii_letters)
    vocabulario.append(palabra)

  for i in range(longitud_texto):
    texto.append(random.choice(vocabulario))

  return texto

texto = generador_lenguaje(1000000, 100000)
print("Texto generado:"," ".join(texto))


# In[ ]:


frecuencia_palabras = Counter(texto)
palabras, frecuencias = zip(*frecuencia_palabras.most_common(50))

plt.figure(figsize=(12, 6))
plt.bar(palabras, frecuencias, color='skyblue')
plt.xticks(rotation=45)
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.title("Frecuencia de las 20 palabras más comunes en el lenguaje artificial")
plt.show()


# In[ ]:


# Instalar en caso de ser necesario
get_ipython().system('pip install prettytable')


# In[ ]:


def contar_frecuencias_y_generar_tabla(texto):
  """
  Cuenta la frecuencia de cada palabra en el texto y genera una tabla de frecuencias.

  Parameters
  ----------
  texto : list
      Lista de palabras en el texto generado.

  Returns
  -------
  PrettyTable
      Tabla que muestra el ranking, la palabra y su frecuencia en orden descendente.
  """
  mapa = {}
  for palabra in texto:
      if palabra in mapa:
          mapa[palabra] += 1
      else:
          mapa[palabra] = 1

  tabla = PrettyTable()
  tabla.field_names = ["Ranking", "Palabra", "Frecuencia"]

  # Ordenar por frecuencia en orden descendente
  palabras_ordenadas = sorted(mapa.items(), key=lambda x: x[1], reverse=True)

  # Agregar a la tabla
  for rank, (palabra, frecuencia) in enumerate(palabras_ordenadas, start=1):
      tabla.add_row([rank, palabra, frecuencia])

  return tabla


# In[ ]:


tabla = contar_frecuencias_y_generar_tabla(texto)
print(tabla)


# ## **Comentarios sobre los resultados obtenidos**
# 
# En un lenguaje natural, existen diversas reglas gramaticales y dependencias que estructuran la semántica. Estas relaciones permiten la formación de estructuras lingüísticas y morfemas, lo que influye en la distribución de las palabras dentro del idioma. La Ley de Zipf establece que en un lenguaje natural la segunda palabra más común aparece aproximadamente la mitad de veces que la más frecuente, y así sucesivamente.
# 
# Sin embargo, en un lenguaje generado de manera completamente aleatoria, no existen tales dependencias gramaticales ni reglas estructurales. Esto significa que las palabras no siguen una distribución jerárquica basada en su función o utilidad dentro del lenguaje, sino que aparencen de forma arbitraria. Lo cual hace poco probable que siga la Ley de Zipf, ya que esta ley surge precisamente de la organización y estructura de los lenguajes naturales o artificiales con reglas bien definidas.
# 
# Además, podríamos incluso cuestionar si un conjunto de símbolos generados aleatoriamente puede considerarse realmente un "lenguaje", dado que carece de sintaxis, semántica y una estructura coherente. No obstante, dentro del contexto de esta tarea, podemos concluir que un lenguaje completamente aleatorio no sigue la Ley de Zipf debido a la ausencia de reglas y dependencias lingüísticas, lo que impide que su distribución de frecuencia refleje la de los lenguajes naturales.
# 
# Como comentario adicional, se observa que curiosamente las palabras de menor longitud aparecen más que aquellas palabras de mayor longitud, siendo algo esperado en la distribución de lenguajes naturales.

# # Segunda parte
# 2. Explorar datasets del sitio Hugging Face y elegir documentos de diferentes dominios en Español (al menos 3). Realizar reconocimiento de entidades nombradas (NER).
# 
# * Pueden utilizar subconjuntos de los datasets encontrados
# * Mostrar resultados del reconocimiento
# * Una distribución de frecuencias de las etiquetas más comunes en cada dominio
# * Comentarios generales del desempeño observado.
# 
# Sugerencias: Spacy, CoreNLP (puede ser cualquier otra herramienta)

# ### Herramienta a utilizar
# 
# En mi caso, se decidió por usar **Spacy**

# In[ ]:


get_ipython().system('python -m spacy download es_core_news_sm')


# In[ ]:


nlp = spacy.load("es_core_news_sm")


# In[ ]:


def generacion_entidades(df):
  """
  Genera entidades nombradas (NER) a partir de un DataFrame de textos.

  Parameters
  ----------
  df : pandas.DataFrame
      DataFrame que contiene una columna 'text' con los textos a analizar.

  Returns
  -------
  tuple
      Una tupla que contiene:
      - Un contador de frecuencias de las etiquetas de las entidades.
      - Una lista de las etiquetas únicas encontradas.
  """
  textos = df['text'].values
  mapa = []

  frecuencia_etiquetas = Counter()

  for text in textos[:100]:
    print("#"*100)
    doc = nlp(text)

    for word in doc.ents:
      frecuencia_etiquetas[word.label_] += 1
      if word.label_ in mapa:
        continue
      else:
        mapa.append(word.label_)

    displacy.render(doc, style="ent")
    print("#"*100,"\n")

  return frecuencia_etiquetas, mapa


def generacion_graficas(frecuencia_etiquetas_1, dominio):
  """
  Genera una gráfica de barras con las frecuencias de las etiquetas de entidades nombradas.

  Parameters
  ----------
  frecuencia_etiquetas_1 : collections.Counter
      Contador de frecuencias de las etiquetas de entidades.
  dominio : str
      Dominio o contexto de los textos analizados.

  Returns
  -------
  None
      Muestra una gráfica de barras con las frecuencias de las etiquetas.
  """
  etiquetas, frecuencias = zip(*frecuencia_etiquetas_1.most_common(10))
  plt.figure(figsize=(10, 5))
  plt.bar(etiquetas, frecuencias, color='lightgreen')
  plt.xlabel("Frecuencia")
  plt.ylabel("Etiqueta")
  plt.title(f"Distribución de frecuencias de etiquetas en NER para el dominio de {dominio}")
  plt.show()


def generacion_graficas_por_categoria(df, frecuencia_etiquetas_1):
    """
    Genera gráficos de barras mostrando la distribución de etiquetas NER encontradas por categoría del dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con las noticias y sus categorías.
    frecuencia_etiquetas_1 : collections.Counter
        Contador con las frecuencias de las etiquetas de entidades.

    Returns
    -------
    None
        Muestra gráficos de barras por cada categoría.
    """
    categorias = df['label'].unique()

    for categoria in categorias:
        subset = df[df['label'] == categoria]
        frecuencia_etiquetas_categoria = Counter()

        for text in subset['text']:
            doc = nlp(text)
            for word in doc.ents:
                frecuencia_etiquetas_categoria[word.label_] += 1

        etiquetas, frecuencias = zip(*frecuencia_etiquetas_categoria.most_common(10)) if frecuencia_etiquetas_categoria else ([], [])

        plt.figure(figsize=(10, 5))
        plt.bar(etiquetas, frecuencias, color='lightblue')
        plt.xlabel("Etiqueta")
        plt.ylabel("Frecuencia")
        plt.title(f"Distribución de etiquetas NER en categoría: {categoria}")
        plt.xticks(rotation=45)
        plt.show()


# Este conjunto de funciones son para poder hacer la generación de gráficas y obtener las entidades encontradas con Spacy.

# ### **Aplicando a titulos de noticias**
# 
# El siguiente conjunto de datos contiene titulos de noticias en español que pertenecen a diferentes temas, desde economía o temas énfocados completamente a colombia.
# 
# Las noticias fueron tomadas de la pagina de RCN y tiene clasificaciones en los titulos:
#       
#       ['salud' 'tecnologia' 'colombia' 'economia' 'deportes']
# 
# Usando las funciones anteriores, se va tomar el dataset, encontrar las entidades usando spacy y posteriormente generar la distribución de las entidades encontradas.
# 
# Se uso el dataset train.

# In[ ]:


splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/Nicky0007/titulos_noticias_rcn_clasificadas/" + splits["train"])

display(df.head(5))

frecuencia_etiquetas_1, mapa_1 = generacion_entidades(df)


# **Significado de las etiquetas encontradas en el texto**

# In[ ]:


# Explicacion de etiquetas encontradas
for i in mapa_1:
  print(f'Abreviacion: {i}   Significado:',spacy.explain(i))


# ### **Comentarios sobre las etiquetas encontradas**
# 
# El análisis de entidades nombradas (NER) en los títulos de noticias utilizando spaCy ha permitido identificar cuatro tipos principales de entidades:
# 
# * LOC (Ubicaciones no-GPE): Sugiere que los títulos incluyen referencias geográficas específicas, posiblemente en noticias de deportes (sedes de eventos) o colombia (accidentes geográficos nacionales).
# 
# * MISC (Misceláneos): Incluye eventos, nacionalidades, productos o nombres de obras de arte. Lo cual puede estar relacionado con noticias de tecnología, economía (marcas o productos) y deportes (torneos o competiciones), cosas de covid y las vacunas.
# 
# * ORG (Organizaciones): Empresas, agencias e instituciones fueron identificadas frecuentemente, lo que sugiere que las noticias tratan sobre actores clave en temas de economía (empresas y bancos), salud (hospitales, farmacéuticas), tecnología (empresas de innovación) y colombia (instituciones gubernamentales).
# 
# * PER (Personas): La mención de personas o familias en los títulos resalta la importancia de figuras públicas en las noticias. Es probable que estas entidades aparezcan con mayor frecuencia en las categorías de deportes (atletas y entrenadores), colombia (políticos y figuras públicas) y tecnología (innovadores y empresarios).

# ## Una distribución de frecuencias de las etiquetas más comunes en cada dominio

# In[ ]:


generacion_graficas(frecuencia_etiquetas_1, "títulos de noticias")


# In[ ]:


generacion_graficas_por_categoria(df,frecuencia_etiquetas_1)


# ### **Comentarios**
# 
# Usando el modelo NER, podemos notar que en el dataset fue encontrado predominan 4 etiquetas, las cuales podemos notar que resultan ser lógicas al tratarse de los títulos de noticias, de manera que:
# 
#     Abreviacion: LOC   Significado: Non-GPE locations, mountain ranges, bodies of water
#     Abreviacion: MISC   Significado: Miscellaneous entities, e.g. events, nationalities, products or works of art
# 
# Tienen sentido, porque normalmente al hablar de noticias siempre se suele mencionar ubicaciones, siendo como un factor que atrae la atención de las personas. Así mismo, mencionar eventos, es demasiado común.
# 
# Es por ello que podemos notar que los títulos de noticias suelen utilizar bastantes nombres de localidades y de eventos o productos para capturar la atención de las personas. Al ser títulos puedo concluir que justamente es una estrategía para capturar la mayor cantidad de visitas a la noticia.
# 
# *Catar 2022 ¡Argentina LOC es campeona del mundo en penales! reviva los goles de la final de Catar 2022 Vea MISC aquí los goles de la final del Mundial de Catar 2022 MISC entre Argentina LOC y Francia LOC.*
# 
# Este es un claro ejemplo de mencionar una localidad, que en este caso son países, para hacer notar que justamente son campeones.
# 
# En general, la distribución de estas entidades refleja la naturaleza de los titulares periodísticos, destacando lugares, organizaciones, eventos y personas relevantes para cada categoría de noticia.
# 
# De manera más detallada, podemos notar que en la distribución de estos etiquetas en las diferentes categorías que tenemos nos da información sobre quén tipo de información suelen tener estas noticias. Aunque, la mayoría de las categorías parecen tener la misma distribución de las etiquetas, siendo diferentes para salud y colombia.
# 

# ### **Aplicando a textos de podcast**
# 
# Este dataset contiene fragmentos de texto extraídos del pódcast "Deforme Semanal", junto con anotaciones que identifican un conjunto específico de entidades.
# 
# La transcripción del audio se realizó inicialmente, seguida de la anotación automática con GPT-3 y una posterior curación utilizando Argilla. El dataset está en español y cubre principalmente temas como el amor, el feminismo y el arte, que son los ejes centrales del pódcast.
# 
# Se uso el dataset train.

# In[ ]:


import pandas as pd

splits = {'train': 'data/train-00000-of-00001-ac966cb4770ad145.parquet', 'test': 'data/test-00000-of-00001-d6a836df2a000de9.parquet'}
df = pd.read_parquet("hf://datasets/somosnlp-hackathon-2023/podcasts-ner-es/" + splits["train"])

display(df)

frecuencia_etiquetas_2, mapa_2 = generacion_entidades(df)


# In[ ]:


# Explicacion de etiquetas encontradas
print("Explicacion de etiquetas encontradas:")
for i in mapa_2:
  print(f'Abreviacion: {i}   Significado:',spacy.explain(i))


# ### **Comentarios sobre las etiquetas encontradas**
# 
# Las etiquetas encontradas permiten categorizar diferentes tipos de entidades mencionadas en los fragmentos de texto:
# 
# * MISC (Misceláneos): Incluye eventos, nacionalidades, productos y obras de arte. Esta etiqueta es particularmente relevante, dado que el pódcast trata temas culturales, mencionando libros, películas, exposiciones y movimientos artísticos.
# 
# * PER (Personas): Se identificaron nombres de figuras relevantes en el ámbito cultural, político y social. Esto es consistente con el formato del pódcast, que suele referirse a personajes históricos y contemporáneos influyentes en temas de feminismo y arte.
# 
# * LOC (Ubicaciones no-GPE): Sugiere que los episodios incluyen menciones a escenarios importantes en el contexto de la conversación.
# 
# * ORG (Organizaciones): Se detectaron empresas, agencias e instituciones, lo que indica que el pódcast menciona organismos culturales, editoriales, medios de comunicación o colectivos feministas.
# 

# ## Una distribución de frecuencias de las etiquetas más comunes en cada dominio

# In[ ]:


generacion_graficas(frecuencia_etiquetas_2, "podcast")


# ### **Comentarios**
# 
# La presencia de diversas etiquetas sugiere que el modelo debe ser capaz de reconocer entidades en un lenguaje fluido y natural, característico de los diálogos en pódcasts.
# 
# También se puede considerar que este dataset es lo más cercano a una conversación natural, aunque es claro que se encuentra sesgado a un tema específico al ser un podcast, pero generalmente las conversaciones son más naturales que otros formatos.
# 
# En este caso, se puede notar que la etiqueta de personas es la que más presencia tiene, lo cual tiene sentido porque al ser un podcast es general que toquen temas relacionadas a personas famosas o lugares importantes, incluso productos o eventos. De manera que podemos notar que estos podcast están más enfocado a tocar temas donde se hace mención de personas, la cual se puede concluir que son personas que tienen cierta relevancia como para mencionarlas en un podcast, porque si no fuera así, la mayoría del público no estaría entendiendo. Aunque también se observa que las conversaciones van enfocadas a contar pequeñas anecdotas, finalmente es un podcast.
# 
# *Y ella, como era una drama, una señora listísima, fantástica, que intelectualizaba todo para Arendt PER , siempre estuvo en segundo plano sus reflexiones sobre el amor, consideraba que el amor no era sobre algo que habría que... Debatir PER.*
# 
# Este es un buen ejhemplo de que justo de habla de alguie que tiene cierta relevancia, por ende se menciona varias veces, además del nombre de la presentadora, etc, lo que hace que esta etiqueta sea tan común. Esto también podría generalizarse que en los podcast generalmente lo que se repite más son nombres. Ya sea para hablar con el invitado o para el dueño del podcast.
# 

# ### **Aplicando a textos de información médica**
# 
# Para este dataset, se maneja que información médica, documentos que pueden abarcar desde conversaciones a datos concretos, pero todo relacionado al mundo de la medicina.
# 
# Se usa el corpus detault

# In[ ]:


import pandas as pd

splits = {'corpus_default': 'AIR-Bench_24.05/default/corpus.jsonl', 'queries_default_dev': 'AIR-Bench_24.05/default/dev_queries.jsonl', 'queries_default_test': 'AIR-Bench_24.05/default/test_queries.jsonl'}
df = pd.read_json("hf://datasets/AIR-Bench/qa_healthcare_es/" + splits["corpus_default"], lines=True)

display(df)

frecuencia_etiquetas_3, mapa_3 = generacion_entidades(df)


# In[ ]:


# Explicacion de etiquetas encontradas
print("Explicacion de etiquetas encontradas:")
for i in mapa_3:
  print(f'Abreviacion: {i}   Significado:',spacy.explain(i))


# ### **Comentarios sobre las etiquetas encontradas**
# 
# * ORG: Aparece porque los textos médicos mencionan instituciones, hospitales y organizaciones clave para estudios y regulaciones.
# 
# * MISC: Surge por términos misceláneos como eventos, fechas o procedimientos médicos que no encajan en otras categorías.
# 
# * PER: Está presente por la mención de investigadores, médicos o pacientes relevantes para casos o descubrimientos.
# 
# * LOC: Se usa para ubicar estudios o casos en regiones, hospitales o áreas geográficas específicas.
# 
# Cada etiqueta refleja elementos esenciales para contextualizar la información médica.

# In[ ]:


generacion_graficas(frecuencia_etiquetas_3, "healthcare")


# ### **Comentarios**
# 
# 1. LOC (1er lugar en frecuencia)
# La etiqueta LOC es la más frecuente porque los textos médicos suelen estar fuertemente vinculados a ubicaciones geográficas o espacios específicos. Esto se debe a que:
# 
#   * Contexto geográfico: Muchos estudios médicos, epidemiológicos o casos clínicos están asociados a regiones, ciudades o países donde se realizan investigaciones o se reportan brotes de enfermedades.
# 
#   * Ubicaciones de instituciones: Se mencionan hospitales, clínicas o centros de investigación, que son clave para entender dónde se llevan a cabo los procedimientos o estudios.
# 
#   * Relevancia en salud pública: La localización es crucial para analizar la distribución de enfermedades, el acceso a tratamientos o la implementación de políticas sanitarias.
# 
#   * Ejemplos comunes:
# 
#     "Uruguay es la principal causa de muerte en..."
# 
# 
# 2. ORG (2do lugar en frecuencia)
# La etiqueta ORG ocupa el segundo lugar porque los textos médicos hacen frecuentes referencias a instituciones y organizaciones, las cuales son pilares en el ámbito de la salud. Esto incluye:
# 
#   * Instituciones de salud: Hospitales, clínicas y laboratorios que realizan investigaciones o brindan tratamientos.
# 
#   * Agencias reguladoras: Organismos que establecen normativas y estándares, como el CECMED o la OMS.
# 
#   * Organizaciones científicas: Entidades que promueven la investigación y colaboración en salud, como CIBERFES o AstraZeneca.
# 
#   Estas menciones son esenciales para validar la credibilidad de la información y entender el contexto institucional detrás de estudios o tratamientos.
# 
#   Ejemplos comunes:
# 
#     "El Complejo Asistencial en Salud Mental Benito Menni..."
# 
# 3. MISC (3er lugar en frecuencia)
# La etiqueta MISC es la tercera más frecuente porque los textos médicos incluyen una variedad de términos misceláneos que no encajan en otras categorías, pero que son relevantes para complementar la información. Estos términos pueden ser:
# 
#   * Eventos médicos: Congresos, jornadas o conferencias donde se presentan avances científicos.
# 
#   * Fechas y periodos: Referencias temporales que contextualizan estudios o casos clínicos.
# 
#   * Productos o procedimientos: Nombres de medicamentos, técnicas o herramientas utilizadas en tratamientos.
# 
#   Ejemplos comunes:
# 
#     "El IX Congreso de Atención Primaria de Castilla-La Mancha..."
# 
# 4. PER (4to lugar en frecuencia)
# La etiqueta PER es la menos frecuente porque, aunque los textos médicos mencionan personas, estas suelen ser menos recurrentes en comparación con ubicaciones, instituciones o términos misceláneos. Las menciones de personas suelen estar relacionadas con:
# 
#   * Investigadores y médicos: Profesionales que lideran estudios o descubrimientos.
# 
#   * Pacientes: Casos clínicos específicos que se usan como ejemplos.
# 
#   * Figuras históricas: Personajes relevantes en la historia de la medicina.
# 
#   Aunque son importantes, las referencias a personas tienden a ser menos numerosas en comparación con otros elementos contextuales.
# 
#   Ejemplos comunes:
# 
#     "Rosa recordó que las enfermedades cardiovasculares son..."
# 
# ---
# 
# ## Comentarios generales
# 
# De manera general, se puede decir que raelmente hace un buen trabajo, aunqe claro que comete fallas y justamente las fallas son las más clásicas, cuando se confunden sustantivos, o cuando no se hace bien la discriminación de verbos y sujetos, etc. Un ejemplo podría ser el siguiente:
# 
# *Claro PER , explica tú siendo Arendt PER , judía, perseguida en Alemania LOC , encarcelada y escapando emigrando a Estados Unidos LOC por el nazismo.¨*
# 
# Donde reconoce Claro como persona, pero es una expresión para describir un escenario durante la narración.
# 
# De manera general podría ser un buen comienzo para comenzar con la extracción de entidades, aunque **claro**, no me quedaría con este modelo. Lo más viable sería justamente usar modelos ajustados, como transformers con ajustados a los datos, que es el proceso de fine tuning a estas tareas específicas.
# 
