import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from loguru import logger


@st.cache_resource
def load_llm(_model="gemma3:4b", _temperature=0.2, _num_thread=2):
    """
    Crea y retorna una instancia del modelo LLM especificado.

    Parameters
    ----------
    _model : str, optional
        Nombre del modelo a cargar. Por defecto es "gemma3:4b".
    _temperature : float, optional
        Valor de temperatura para el muestreo del modelo. Por defecto es 0.2.
    _num_thread : int, optional
        Número de hilos para la ejecución del modelo. Por defecto es 2.

    Returns
    -------
    OllamaLLM
        Instancia configurada del modelo LLM.
    """
    logger.info(f"Solicitando instancia de LLM: {_model}")
    return OllamaLLM(model=_model, temperature=_temperature,
                     num_thread=_num_thread)


def create_qa_chain(llm_instance, retriever_instance):
    """
    Crea una cadena de preguntas y respuestas usando un LLM y un recuperador.

    Parameters
    ----------
    llm_instance : BaseLanguageModel
        Instancia del modelo de lenguaje a utilizar.
    retriever_instance : BaseRetriever
        Instancia del recuperador de documentos.

    Returns
    -------
    RetrievalQA or None
        Cadena de QA configurada, o None si faltan componentes.
    """
    if llm_instance and retriever_instance:
        logger.debug("Creando cadena de RetrievalQA")
        return RetrievalQA.from_chain_type(llm=llm_instance,
                                           chain_type="stuff",
                                           retriever=retriever_instance)
    logger.warning(
        "No se pudo crear la cadena de QA, LLM o retriever no disponible."
    )
    return None
