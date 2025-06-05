# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: transformers_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Alumnos:
#
# * ### Vázquez Martínez Fredin Alberto

# %% [markdown]
# # Aplicación: StudyBuddy
# Aplicación que con base en tus notas de clase y documentos relacionados te ayuda a estudiar para pasar tu examen final.
#
# ## Desarrollo
#
# ### Input:
# Se va a considerar un agente con langchain capaz de leer notas, la entrada esperada serán Pdfs, posteriormente va captar información.
#
#
# ### Output:
# La salida esperada será un resumen de puntos más importantes de tus notas.
#
# En caso que el usuario incluya alguna de los siguientes puntos, se va esperar resultados adicionales:
#
# 1. Palabras clave o temas clave: además del resumen general, el agente será capaz de poder enfocar el resumen de las notas para ese tema solicitado por el usuario. 
#
# 2. Preguntas: el agente tendrá la capacidad de responder a preguntas en base a las notas dadas.

# %%
## Instalando las dependencias ##
# !pip install langchain_ollama
# !pip install langchain_community
# !pip install pymupdf
# !pip install langchain_text_splitters
# !pip install langchain_chroma
# !pip install langgraph

# %%
## Cargando las librerias ##
import os 
from rich import print as rprint
from rich.rule import Rule
from langchain_ollama.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Markdown, display
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


# loaders
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader, UnstructuredPDFLoader

# langGraph
from langgraph.graph import MessagesState, StateGraph
from IPython.display import Image, display
from langgraph.graph import START, StateGraph

# %% [markdown]
# ### Cargando el modelo y definiendo loaders para recuperación de información

# %%
## Cargando el modelo ##
MODEL = "llama3.2"
llm = ChatOllama(model=MODEL,base_url="http://172.29.128.1:11434")

## Definiendo loaders ##

name_dir = 'pdfs_notes'
documents = [
    doc
    for file in os.listdir(name_dir)
    if file.endswith(".pdf")
    for doc in PyMuPDFLoader(os.path.join(name_dir, file)).load()
]

rprint(f"Total documents {len(documents)}")
for i, doc in enumerate(documents[:3], start=1):
    rprint(Rule(f"Doc {i}"))
    rprint(doc.page_content[:300])

# %%
## Separando la informacion de los documentos para una mejor recuperación de la información ##
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(documents)

rprint(f"Splits {len(all_splits)}")
for i, split in enumerate(all_splits[30:33], start=1):
    rprint(Rule(f"split #{i}"))
    rprint(split)

# %% [markdown]
# ### Cargando modelo de embeddings
#
# Los embeddings serán usados para poder hacer búsqueda en los textos que fueron dados, usando similitud de coseno. Estos embeddings serán usados para crear la base de datos de vectores, de esta manera podemos agregar esos vectores de la información que tenemos para poder hacer queries.

# %%
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest",
    base_url="http://172.29.128.1:11434")
vector_store = Chroma(
    # Nombramos nuestra colección
    collection_name="vector_BD",
    embedding_function=embeddings,
    persist_directory="./documents_vector_db"
)

# Indexando nuestra información
ids = vector_store.add_documents(documents=all_splits)
rprint(ids[:10])

# %% [markdown]
# Ya con esta base de datos de vectores de nuestra información, es posible hacer consultas de similitud, con similitud coseno, para encontrar splits de los documentos cargados, que tengan mayor similitud con nuestra búsqueda.

# %%
# Búsqueda de término
results = await vector_store.asimilarity_search("Sobreajuste")
rprint(results[0])

# Búsqueda de pregunta
embedding = embeddings.embed_query("por qué se produce el sobreajuste?")
results = vector_store.similarity_search_by_vector(embedding)
rprint(f"Results={len(results)}")
for i, result in enumerate(results):
    rprint(Rule(f"Result #{i}"))
    rprint(result.page_content)

# %% [markdown]
# ## ¿Qué tenemos hasta el momento?
#
# 1. Se cargan los documentos usar
# 2. Se separan en bloques para poder recuperar de mejor manejar la información de nuestros documentos
# 3. Crear la base de datos de vectores
# 4. Poder hacer búsqueda de similitud coseno por medio de embeddings
#
# ## Creación del prompt

# %%
PROMPT_TEMPLATE = """
### Rol y Objetivo:
Eres StudyBuddy, un asistente de estudio especializado en transformar notas académicas en material de aprendizaje efectivo. 
Tu principal objetivo es ayudar al usuario a preparar su examen final mediante técnicas de estudio comprobadas.

### Instrucciones de Contexto:
1. **Base de conocimiento**: Usa EXCLUSIVAMENTE la información del contexto proporcionado (vectorizado de las notas del usuario)
2. **Limitaciones**: Si la pregunta no está cubierta en el contexto o es ambigua:
   - Responde: "No encuentro esa información en tus notas. ¿Quieres que profundice en otro tema?"
   - Nunca inventes información o conceptos no presentes en el contexto

### Funcionalidades Principales:
#### 1. Generación de Resúmenes:
   - Crea resúmenes estructurados usando:
     * Técnica de Cornell (conceptos clave + notas laterales + resumen final)
     * Mapas mentales en formato markdown
     * Diagramas de flujo conceptuales (usando pseudocódigo ASCII)

#### 2. Explicación de Temas:
   - Desglosa conceptos complejos en componentes simples
   - Proporciona analogías prácticas relacionadas con el campo de estudio
   - Incluye ejemplos concretos extraídos del contexto

#### 3. Generación de Prácticas:
   - Crea preguntas de práctica según tipo:
     * [V/F] Para verificar comprensión básica
     * [Caso práctico] Escenarios aplicados al material
     * [Desarrollo] Preguntas de análisis profundo
   - Incluye rúbricas de evaluación automática para cada respuesta

### Formato de Respuesta:
- Usa markdown para estructurar el contenido
- Prioriza viñetas y listas numeradas
- Incluye etiquetas de dificultad: [BÁSICO|INTERMEDIO|AVANZADO]
- Mantén máximo 3 párrafos por sección

### Parámetros Actuales:
Usuario solicita: {question}
Contexto disponible: {context}

Respuesta:
"""


prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# %%
example_messages = prompt.invoke(
    {"context": "[blue]<My retrieved and absolutely relevant documents>[/]", "question": "[green]<The question in question>[/]"}
).to_messages()

rprint(example_messages[0].content)

# %% [markdown]
# ## Probando el modelo

# %%
# 1. Solicitud del estudiante
question = input(f"StudyBuddy [{MODEL}]>> ")
rprint("1")
# 2. Obtener documentos reelevantes
retrieved_docs = vector_store.similarity_search(question)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
rprint("2")
# 3. Pasarlos a junto con la pregunta al modelo
prompt_result = prompt.invoke({"question": question, "context": docs_content})
rprint("3")
# 4. Generar una respuesta
answer = llm.invoke(prompt_result)

rprint(f'Usuario: {question}')

display(Markdown('**StudyBuddy**: '+answer.content))


# %% [markdown]
# ## Orquestando el flujo del agente con LangGraph

# %%
class State(TypedDict):
    """Define the states of the app"""
    question: str
    context: list[Document]
    answer: str


# %%
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


## Creacion del grafo ##
graph_builder = StateGraph(State).add_sequence([retrieve, generate])

graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

## Grafo ##
display(Image(graph.get_graph().draw_mermaid_png())) 

# %% [markdown]
# ### Flujo del agente con LangGraph
# El siguiente flujo ya muestra como se va generando la respuesta usando LangGraph, en este caso es interesante ya que podemos generar el flujo como mensaje.
#
# Para este caso sea más ilustrativo, desde un notebook se puede usar el código de update_display para poder actualizar la salida y poco a poco ir viendo la respuesta que nos da el agente.

# %%
from IPython.display import display, Markdown, update_display

streamed_response = ""
display_id = "stream"

display(Markdown(""), display_id=display_id)

for message, metadata in graph.stream(
    {"question": "Dame los puntos más probables a preguntar en el examen"},
    stream_mode="messages"
):
    streamed_response += message.content
    update_display(Markdown(streamed_response), display_id=display_id)

# %% [markdown]
# ## Haciendo el modelo conversacional con LangGraph
#
# Esto es tener guardado los mensajes anterior, o sea un historial de mensajes para poder crear el modelo conversacional. Esta funcionalidad ya lo provee langGraph.

# %%
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

# %% [markdown]
# Usamos el decorador
#         
#         @tool(response_format="content_and_artifact")
#
# Para poder indicar que esta es una herramienta que se va usar dentro de un sistema, en este caso dentro del grafo definido anteriormente con langGraph

# %%
graph_msg_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve_tool(query: str):
    """Retrieve information related to a user query"""
    retrieved_docs = vector_store.similarity_search(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


## Ahora, hacemos la llamada de la tool para poder obtener los docs
## más importantes, y después obtener la respuesta.

def query_or_respond(state: MessagesState):
    """Genera una tool call para retrieval o responde directo
    """
    llm_with_tools = llm.bind_tools([retrieve_tool]) # agrega esa herramienta al modelo
    response = llm_with_tools.invoke(state["messages"]) # usando esa herramienta es que ya hace la 
    return {"messages": [response]}


## Funcion para generar la respuesta usando las tools y el prompt 
def generate(state: MessagesState):
    """Genera una respuesta"""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if not message.type == "tool":
            break
        recent_tool_messages.append(message)
    # Obtenemos los mensajes de tools en orden inverso
    tool_messages = recent_tool_messages[::-1]
    # Creando un prompt con los mensajes
    docs_content = "\n\n".join(doc.content for doc in tool_messages)


    system_message = (
                    """
                    ### Rol y Objetivo:
                    Eres StudyBuddy, un asistente de estudio especializado en transformar notas académicas en material de aprendizaje efectivo. 
                    Tu principal objetivo es ayudar al usuario a preparar su examen final mediante técnicas de estudio comprobadas.

                    ### Instrucciones de Contexto:
                    1. **Base de conocimiento**: Usa EXCLUSIVAMENTE la información del contexto proporcionado (vectorizado de las notas del usuario)
                    2. **Limitaciones**: Si la pregunta no está cubierta en el contexto o es ambigua:
                    - Responde: "No encuentro esa información en tus notas. ¿Quieres que profundice en otro tema?"
                    - Nunca inventes información o conceptos no presentes en el contexto

                    ### Funcionalidades Principales:
                    #### 1. Generación de Resúmenes:
                    - Crea resúmenes estructurados usando:
                        * Técnica de Cornell (conceptos clave + notas laterales + resumen final)
                        * Mapas mentales en formato markdown
                        * Diagramas de flujo conceptuales (usando pseudocódigo ASCII)

                    #### 2. Explicación de Temas:
                    - Desglosa conceptos complejos en componentes simples
                    - Proporciona analogías prácticas relacionadas con el campo de estudio
                    - Incluye ejemplos concretos extraídos del contexto

                    #### 3. Generación de Prácticas:
                    - Crea preguntas de práctica según tipo:
                        * [V/F] Para verificar comprensión básica
                        * [Caso práctico] Escenarios aplicados al material
                        * [Desarrollo] Preguntas de análisis profundo
                    - Incluye rúbricas de evaluación automática para cada respuesta

                    ### Formato de Respuesta:
                    - Usa markdown para estructurar el contenido
                    - Prioriza viñetas y listas numeradas
                    - Incluye etiquetas de dificultad: [BÁSICO|INTERMEDIO|AVANZADO]
                    - Mantén máximo 3 párrafos por sección
                    """
        f"{docs_content}"
    )
    
    convertation = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message)] + convertation

    # Run!
    response = llm.invoke(prompt)
    return {"messages": [response]}


# %%
tools = ToolNode([retrieve_tool])
graph_msg_builder.add_node(query_or_respond)
graph_msg_builder.add_node(tools)
graph_msg_builder.add_node(generate)

graph_msg_builder.set_entry_point("query_or_respond")
graph_msg_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"}
)

graph_msg_builder.add_edge("tools", "generate")
graph_msg_builder.add_edge("generate", END)

graph_tools = graph_msg_builder.compile()


# %%
display(Image(graph_tools.get_graph().draw_mermaid_png()))

# %%
input_message = "Dame los puntos mas probables que estén en el examen"

for step, metadata in graph_tools.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="messages",
):
    if step.type == "AIMessageChunk":
        print(step.content, end="")



# %%
input_message = "cual seria el segundo punto"

for step, metadata in graph_tools.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="messages",
):
    if step.type == "AIMessageChunk":
        print(step.content, end="")



# %%
