# Lab of Selected Themes on Computer Linguistics 2025-2

Repositorio para las prácticas de la materia Temas Selectos de Lingüística
Computacional en la Lic. de Ciencias de Datos, IIMAS, UNAM

## Instalación de dependencias

Usaremos `uv` como gestor de dependencias.

- [Instalación del programa](https://docs.astral.sh/uv/getting-started/installation/)

Una ves instalado basta con ejecutar el siguiente comando para levantar jupyter lab local con las dependencias:

```shell
$ uv run jupyter-lab
```

Si precisas instalar alguna dependencia puedes agregarla con:

```shell
$ uv add <dep>
```

**Toma en cuenta que si agregas dependencias estas se agregará al repositorio
principal (`pyproject.toml` y `uv.lock`) con tu PR asi que cuida lo que
precises instalar**

# Prácticas

### Lineamientos generales

- Es muy recomendable entregar las prácticas ya que representa un porcentaje
importante de su calificación (40%) 🤓
- Se dará ~2 semanas para entregar ejercicios (dependiendo de la práctica)
    - En caso de **entregas tardías** abrá una penalización `-1 punto` por cada día
- Si utilizas LLMs, o herramientas generativas reportalos en tus prácticas 🧙
  - Les recomendamos ampliamente que lo intenten por su cuenta primero, es una
  oportunidad de enfrentarse a cosas nuevas y de pensar en soluciones por su
  cuenta :)

## 0. Creación de carpeta personal via *Pull Request (PR)*

Crear un PR con lo siguiente:

- Una carpeta con su username de GitHub dentro de `practicas/` y otra carpeta interna llamada `P0/`
    - `practicas/umoqnier/P0`
- Agrega un archivo llamado `README.md` a la carpeta `P0/` con información básica sobre tí. Ejemplo:
    - `practices/umoqnier/P0/README.md`
    - Usar lenguaje de marcado [Markdown](https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

```markdown
$ cat README.md

# Diego Alberto Barriga Martínez

- Número de cuenta: `XXXXXXXX`
- User de Github: @umoqnier
- Me gusta que me llamen: Dieguito

## Pasatiempos

- Andar en bici

## Proyectos en los que he participado y que me enorgullesen 🖤

- [Esquite](https://github.com/ElotlMX/Esquite/)
```

## 1. Niveles Lingüísticos

### FECHA DE ENTREGA: 16 de Febrero 2025 at 11:59pm

### Fonética

1. Si tenemos un sistema de búsqueda que recibe una palabra ortográfica y devuelve sus transcripciones fonológicas, proponga una solución para los casos en que la palabra buscada no se encuentra en el lexicón/diccionario. *¿Cómo devolver o aproximar su transcripción fonológica?*
  - Reutiliza el sistema de búsqueda visto en clase y mejoralo con esta funcionalidad

### Morfología

2. Obtenga los datos de `test` y `dev` para todas las lenguas disponibles en el Shared Task SIGMORPHON 2022 y haga lo siguiente:
    - En un plot de 4 columnas y 2 rows muestre las siguientes distribuciones (un subplot por lengua):
        - Plot 1: distribución de longitud de palabras
        - Plot 2: distribución de la cuenta de morfemas
        - Plot 3: distribución de categorias (si existe para la lengua)
    - Realice una función que imprima por cada lengua lo siguiente:
        - Total de palabras
        - La longitud de palabra promedio
        - La cuenta de morfemas promedio
        - La categoría más común
    - Con base en esta información elabore una conclusión lingüística sobre la morfología de las lenguas analizadas.
    
### EXTRA:

- Imprimir la [matríz de confusión](https://en.wikipedia.org/wiki/Confusion_matrix) para el etiquetador CRFs visto en clase y elaborar una conclusión sobre los resultados

## 2: Propiedades estadísticas de la lengua

### Fecha de entrega: 2 de Marzo de 2025 11:59pm

1. Verificar si la ley de Zipf se cumple en un lenguaje artificial creado por ustedes.
    - *Ejemplo:* Un "lenguaje artificial" podría ser simplemente un texto donde las secuencias de caracteres fueron generadas aleatoriamente.
2. Explorar `datasets` del sitio [Hugging Face](https://huggingface.co/datasets) y elegir documentos de diferentes dominios en Español (al menos 3). Realizar reconocimiento de entidades nombradas (NER).
    - Pueden utilizar subconjuntos de los datasets encontrados
    - Mostrar resultados del reconocimiento
    - Una distribución de frecuencias de las etiquetas más comunes en cada dominio
    - Comentarios generales del desempeño observado.

*Sugerencias: Spacy, CoreNLP (puede ser cualquier otra herramienta)*

## 3. Práctica: Vectores a palabras

### Fecha de entrega: 19 de Marzo de 2025 @ 11:59pm

Obtenga la matriz de co-ocurrencia para un corpus en español y realice los siguientes calculos:
- Las probabilidades conjuntas
$$p(w_i,w_j) = \frac{c_{i,j}}{\sum_i \sum_j c_{i,j}}$$
- Las probabilidades marginales
$$p(w_i) = \sum_j p(w_i,w_j)$$
- Positive Point Wise Mutual Information (PPMI):
$$PPMI(w_i,w_j) = \max\{0, \log_2 \frac{p(w_i,w_j)}{p(w_i)p(w_j)}\}$$

**Comparación de representaciones**

Aplica reducción de dimensionalidad (a 2D) de los vectores de la matríz con PPMI y de los vectores entrenados en español:

- Realiza un plot de 100 vectores aleatorios (que esten tanto en la matríz como en los vectores entrenados)
- Compara los resultados de los plots:
    - ¿Qué representación dirías que captura mejor relaciones semánticas?
    - Realiza un cuadro comparativo de ambos métodos con ventajas/desventajas

## Práctica 4: Modelos del Lenguaje Neuronales

**Fecha de entrega: 6 de abril de 2025 11:59pm**

A partir del modelo entrenado:

- Sacar los embeddings de las palabras del vocabulario

- Visualizar en 2D los embeddings de algunas palabras (quizá las más frecuentes, excluyendo stopwords)

- Seleccione algunas palabras y verifique sí realmente codifican nociones semánticas, e,g, similitud semántica con similitud coseno entre dos vectores, analogías por medios de operaciones de vectores

**NOTA**: Puedes entrenar el modelo replicando la ejecución del notebook o encontrar el modelo entrenado en la [carpeta de drive](https://drive.google.com/drive/folders/1Mq-UA0ct5iTp-7h8-SxJxwyjdMHXmwO4?usp=drive_link)

### Extra (0.5 pts):

- Correr el modelo de Bengio pero aplicando una técnica de subword tokenization al corpus y hacer generación del lenguaje

- La generación del lenguaje debe ser secuencias de palabras (no subwords)

## Práctica 5: Tech evolution. Caso *POS Tagging*

**Fecha de entrega: 13 de Abril 2025 11:59pm**

- Obten los embeddings de 100 palabras al azar del modelo RNN visto en clase
  - Pueden ser los embeddings estáticos o los dinámicos del modelo
- Aplica un algoritmo de clusterización a las palabras y plotearlas en 2D
  - Aplica algun color para los diferentes clusters
- Agrega al plot los embeddings de las etiquetas POS
  - Utiliza un marcador que las distinga claramente de las palabras
- Realiza una conclusión sobre los resultados observados

### Extra: 0.5pt

- Implementa una red *Long short-term memory units (LSTM)* para la tarea de etiquetado POS
- Reporta el accuracy y comparalo con los resultados de la RNN simple
- Realiza un comentario sobre como impacta la arquitectura LSTM sobre el resultado obtenido
