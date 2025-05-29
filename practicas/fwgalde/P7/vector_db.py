import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from loguru import logger


@st.cache_resource
def load_embeddings(model_name="all-MiniLM-L6-v2"):
    """
    Carga un modelo de embeddings de Hugging Face y lo cachea.

    Parameters
    ----------
    model_name : str, optional
        Nombre del modelo de embeddings a cargar. Por defecto es
        "all-MiniLM-L6-v2".

    Returns
    -------
    HuggingFaceEmbeddings
        Instancia del modelo de embeddings cargado.
    """
    logger.info(f"Solicitando modelo de embeddings: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource
def load_vectordb(persist_dir, _emb):
    """
    Carga una base de vectores ChromaDB desde un directorio persistente.

    Parameters
    ----------
    persist_dir : str
        Ruta al directorio donde se almacenan los vectores persistentes.
    _emb : Embeddings
        Función de embeddings para convertir texto a vectores.

    Returns
    -------
    Chroma
        Instancia de la base de vectores ChromaDB cargada.
    """
    logger.info(f"Solicitando instancia de ChromaDB desde: {persist_dir}")
    return Chroma(persist_directory=persist_dir, embedding_function=_emb)


def add_documents_to_store(vector_store_instance, chunks_to_add):
    """
    Añade fragmentos de documentos a una instancia de base vectorial.

    Parameters
    ----------
    vector_store_instance : Chroma
        Instancia de la base vectorial donde se añadirán los documentos.
    chunks_to_add : list of Document
        Fragmentos de documentos a añadir al almacén vectorial.

    Returns
    -------
    int
        Número de documentos añadidos exitosamente.
    str or None
        Mensaje de error en caso de fallo, o None si no hubo error.
    """
    if not chunks_to_add:
        logger.info(f"No hay nuevos fragmentos para añadir a ChromaDB.")
        return 0, None

    try:
        logger.info(f"Añadiendo {len(chunks_to_add)} fragmentos a ChromaDB.")
        vector_store_instance.add_documents(chunks_to_add)
        return len(chunks_to_add), None

    except Exception as e:
        logger.exception("Error al añadir documentos.")
        return 0, f"Error al añadir a Chroma: {e}"


def get_retriever(vector_store_instance, k_results=3):
    """
    Obtiene un recuperador basado en una instancia de base vectorial.

    Parameters
    ----------
    vector_store_instance : Chroma
        Instancia de la base vectorial desde la cual obtener el recuperador.
    k_results : int, optional
        Número de resultados más relevantes a recuperar. Por defecto es 3.

    Returns
    -------
    BaseRetriever or None
        Recuperador configurado, o None si no hay base vectorial disponible.
    """
    if vector_store_instance:
        return vector_store_instance.as_retriever(
            search_kwargs={"k": k_results}
        )
    return None


def get_store_collection_count(vector_store_instance):
    """
    Obtiene el número de documentos en la colección de la base vectorial.

    Parameters
    ----------
    vector_store_instance : Chroma
        Instancia de la base vectorial.

    Returns
    -------
    int
        Cantidad de documentos almacenados en la colección. 0 si está vacía
        o no accesible.
    """
    if vector_store_instance._collection.count():
        return vector_store_instance._collection.count()
    return 0
