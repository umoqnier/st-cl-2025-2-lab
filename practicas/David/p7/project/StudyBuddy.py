import os 
from rich import print as rprint
from rich.rule import Rule
from langchain_ollama.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Markdown, display, Image
from langchain_core.documents import Document
from typing import List, Tuple
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessageChunk
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader, UnstructuredPDFLoader

class StudyBuddyAssistant:
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        embedding_model: str = "nomic-embed-text:latest",
        documents_dir: str = "./pdfs_notes",
        persist_directory: str = "./documents_vector_db"
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.documents_dir = documents_dir
        self.persist_directory = persist_directory

        self.llm = ChatOllama(model=self.model_name)
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = None
        self.graph = None

        self._load_documents()
        self._setup_graph()

    def _load_documents(self):
        documents = [
            doc
            for file in os.listdir(self.documents_dir)
            if file.endswith(".pdf")
            for doc in PyMuPDFLoader(os.path.join(self.documents_dir, file)).load()
        ]
        rprint(f"Total documents loaded: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            add_start_index=True
        )
        self.all_splits = text_splitter.split_documents(documents)
        rprint(f"Total splits created: {len(self.all_splits)}")

        self.vector_store = Chroma(
            collection_name="vector_BD",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.vector_store.add_documents(documents=self.all_splits)

    def add_documents(self, file_paths: List[str]):
        """"
          Añade documentos a la biblioteca de estudio.
          Args:
            file_paths (List[str]): Ruta al archivo PDF que se desea añadir.
          Returns:
            None
        """
        documents = []
        for file in file_paths:
            if file.endswith(".pdf"):
                documents.extend(PyMuPDFLoader(file).load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(documents)
        self.vector_store.add_documents(splits)

    def _retrieve_tool(self, query: str) -> Tuple[str, List[Document]]:
        """
        Recupera documentos relevantes del vector store en función de la consulta del usuario.
        Devuelve una cadena serializada con el contenido y la lista de documentos relevantes.
        """
        retrieved_docs = self.vector_store.similarity_search(query)
        if not retrieved_docs:
            return "No encuentro esa información en tus notas. ¿Quieres que profundice en otro tema?", []
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def _build_system_prompt(self, context: str) -> str:
        return f"""
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

        ### Formato de Respuesta:
        - Usa markdown para estructurar el contenido
        - Prioriza viñetas y listas numeradas
        - Mantén máximo 3 párrafos por sección
        
        {context}
        """

    def _query_or_respond(self, state):
        """
           Procesa el estado para responder a una pregunta o generar un resumen.
           Si el estado contiene una pregunta, la responde. Si no, genera un resumen.
           El estado debe contener al menos una pregunta o una lista de documentos.
        """
    
        llm_with_tools = self.llm.bind_tools([self._retrieve_tool])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _generate_response(self, state):
        """
           Genera una respuesta basada en el contexto del estado.
           El estado debe contener al menos un documento o una pregunta.
        """
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if not getattr(message, "type", None) == "tool":
                break
            recent_tool_messages.append(message)

        tool_messages = recent_tool_messages[::-1]
        docs_content = "\n\n".join(getattr(doc, "content", getattr(doc, "page_content", "")) for doc in tool_messages)
        system_message = SystemMessage(self._build_system_prompt(docs_content))

        conversation = [
            message
            for message in state["messages"]
            if getattr(message, "type", None) in ("human", "system")
            or (getattr(message, "type", None) == "ai" and not getattr(message, "tool_calls", None))
        ]

        prompt = [system_message] + conversation
        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def _setup_graph(self):
        """
           Configura el grafo de la aplicación.
           El grafo debe tener los nodos necesarios para manejar la interacción con el usuario.
           Los nodos deben estar conectados de manera adecuada para permitir la transición entre diferentes partes del proceso.
           El grafo debe ser configurado para permitir la transición entre diferentes partes del proceso. Por ejemplo, si el usuario pregunta una pregunta, se debe conectar ese nodo con un nodo que maneje la respuesta a esa pregunta. Si el usuario solicita herramientas, se debe conectar ese nodo con un nodo que maneje las herramientas disponibles.
        """
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", self._query_or_respond)
        graph_builder.add_node("tools", ToolNode([self._retrieve_tool]))
        graph_builder.add_node("generate", self._generate_response)
        graph_builder.set_entry_point("query_or_respond")

        def tools_condition(state):
            last_msg = state["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                return "tools"
            return END

        graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        self.graph = graph_builder.compile()

    def ask(self, question: str, stream: bool = True):
        """
            Realiza una pregunta al asistente y devuelve la respuesta.
            :param question: La pregunta que se desea realizar.
            :param stream: Si es True, el asistente proporcionará respuestas paso a paso. Si es False, se espera una respuesta completa.
        """
        inputs = {"messages": [{"role": "user", "content": question}]}
        full_response = ""
        for step, metadata in self.graph.stream(inputs, stream_mode="messages"):
            if step.type == "AIMessageChunk":
                if stream:
                    print(step.content, end="")
                full_response += step.content
        return full_response
