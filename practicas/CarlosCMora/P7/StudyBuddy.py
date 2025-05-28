############################
## Cargando las librerias ##
############################
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

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessageChunk
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition


##########################
## Agente de StudyBuddy ##
##########################

class StudyBuddyAssistant:
    def __init__(
        self,
        model_name: str = "llama3.2",
        embedding_model: str = "nomic-embed-text:latest",
        documents_dir: str = "pdfs_notes",
        persist_directory: str = "./documents_vector_db"
    ):
        """
        Inicializa el asistente de estudio con configuración personalizable.
        
        Args:
            model_name: Nombre del modelo LLM a usar (por defecto: "llama3.2")
            embedding_model: Modelo de embeddings a usar (por defecto: "nomic-embed-text:latest")
            documents_dir: Directorio con los documentos PDF de notas
            persist_directory: Directorio para almacenar la base de datos vectorial
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.documents_dir = documents_dir
        self.persist_directory = persist_directory

        # Inicializar componentes
        self.llm = ChatOllama(model=self.model_name)
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = None
        self.graph = None


    
    def _load_documents(self):
        """Carga y procesa los documentos PDF del directorio especificado."""
        # Cargar documentos PDF
        documents = [
            doc
            for file in os.listdir(self.documents_dir)
            if file.endswith(".pdf")
            for doc in PyMuPDFLoader(os.path.join(self.documents_dir, file)).load()
        ]
        
        rprint(f"Total documents loaded: {len(documents)}")
        
        # Dividir documentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            add_start_index=True
        )
        self.all_splits = text_splitter.split_documents(documents)
        
        rprint(f"Total splits created: {len(self.all_splits)}")
        
        # Crear vector store
        self.vector_store = Chroma(
            collection_name="vector_BD",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        
        # Indexar documentos
        self.vector_store.add_documents(documents=self.all_splits)


    def _retrieve_tool(self, query: str):
        """
        Herramienta para recuperar información relevante de la base de conocimientos.
        """
        retrieved_docs = self.vector_store.similarity_search(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    

    def _query_or_respond(self, state):
        """
        Decide si hacer una llamada a herramienta o responder directamente.
        """
        llm_with_tools = self.llm.bind_tools([self._retrieve_tool])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    

    def _generate_response(self, state):
        """
        Genera una respuesta usando el contexto recuperado.
        """
        # Obtener mensajes de herramientas
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if not message.type == "tool":
                break
            recent_tool_messages.append(message)
        
        tool_messages = recent_tool_messages[::-1]
        docs_content = "\n\n".join(doc.content for doc in tool_messages)

        # Crear system message
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
        response = self.llm.invoke(prompt)
        return {"messages": [response]}
    

    def _setup_graph(self):
        """Configura el grafo de flujo de trabajo."""
        graph_builder = StateGraph(MessagesState)
        
        # Añadir nodos
        graph_builder.add_node("query_or_respond", self._query_or_respond)
        graph_builder.add_node("tools", ToolNode([self._retrieve_tool]))
        graph_builder.add_node("generate", self._generate_response)
        
        # Configurar flujo
        graph_builder.set_entry_point("query_or_respond")
        
        def tools_condition(state):
            last_msg = state["messages"][-1]
            if last_msg.tool_calls:
                return "tools"
            return END
        
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"}
        )
        
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        
        self.graph = graph_builder.compile()


    def ask(self, question: str, stream: bool = True):
        """
        Realiza una pregunta al asistente.
        
        Args:
            question: La pregunta del usuario
            stream: Si True, muestra la respuesta en streaming
            
        Returns:
            La respuesta completa del asistente
        """
        inputs = {"messages": [{"role": "user", "content": question}]}
        self._load_documents()
        self._setup_graph()
        full_response = ""
        for step, metadata in self.graph.stream(
           inputs,
            stream_mode="messages",
        ):
            if step.type == "AIMessageChunk":
                print(step.content, end="")
                yield step.content
        return 

def ask_stream(self, question: str):
    """Versión generadora para streaming"""
    inputs = {"messages": [{"role": "user", "content": question}]}
    self._load_documents()
    self._setup_graph()

    full_response = ""
    for step, metadata in self.graph.stream(
        inputs,
        stream_mode="messages",
    ):
        if step.type == "AIMessageChunk":
            full_response += step.content
            yield full_response
