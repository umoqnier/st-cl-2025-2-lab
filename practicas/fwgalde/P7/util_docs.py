import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

def ensure_dirs_exists(paths):
    """
    Crea los directorios especificados si no existen.

    Parameters
    ----------
    paths : list of str
        Lista de rutas de directorios a asegurar.

    Returns
    -------
    None
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Directorio asegurado {path}")


def save_uploaded_files(uploaded_files, save_path):
    """
    Guarda archivos subidos en el directorio especificado.

    Parameters
    ----------
    uploaded_files : list of UploadedFile
        Lista de archivos subidos a guardar.
    save_path : str
        Ruta del directorio donde se guardarán los archivos.

    Returns
    -------
    saved_file_paths : list of str
        Rutas completas de los archivos guardados exitosamente.
    feedback_messages : list of str
        Mensajes de confirmación o error por cada archivo procesado.
    """
    saved_file_paths = []
    feedback_messages = []

    if not uploaded_files:
        return saved_file_paths, feedback_messages

    for uf in uploaded_files:
        file_path = os.path.join(save_path, uf.name)

        try:
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
            saved_file_paths.append(file_path)
            feedback_messages.append(f"✅ {uf.name}: guardado")
            logger.info(f"Archivo guardado: {file_path}")
        except Exception as e:
            feedback_messages.append(f"❌ {uf.name}: error al guardar - {e}")
            logger.exception(f"Error al guardar {uf.name}")
    return saved_file_paths, feedback_messages


def load_docs_from_files_paths(list_of_file_paths):
    """
    Carga documentos desde rutas de archivos .txt o .pdf soportados.

    Parameters
    ----------
    list_of_file_paths : list of str
        Lista de rutas de archivos a procesar.

    Returns
    -------
    langchain_documents : list of Document
        Documentos cargados con contenido válido.
    feedback_messages : list of str
        Mensajes sobre el estado de carga de cada archivo.
    """
    langchain_documents = []
    feedback_messages = []

    for file_path in list_of_file_paths:
        file_name = os.path.basename(file_path)

        try:
            if file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                feedback_messages.append(
                    f"⚠️ {file_name}: formato no soportado (omitido)"
                )
                logger.warning(f"Formato no soportado omitido: {file_path}")
                continue

            loaded_docs_for_file = loader.load()
            valid_docs = []
            for doc in loaded_docs_for_file:
                if doc.page_content and doc.page_content.strip():
                    valid_docs.append(doc)
            if valid_docs:
                langchain_documents.extend(valid_docs)
                feedback_messages.append(
                    f"📄 {file_name}: contenido cargado ({len(valid_docs)} "
                    "parte(s).)"
                )
                logger.info(f"Cargado contenido de {file_path}")
            else:
                feedback_messages.append(
                    f"⚠️ {file_name}: sin contenido útil o vacío"
                )
                logger.warning(f"Sin contenido útil en {file_path}")

        except Exception as e:
            feedback_messages.append(f"❌ {file_name}: error al cargar - {e}")
            logger.exception(f"Error al cargar {file_path}")
    return langchain_documents, feedback_messages


def split_langchain_docs(docs_to_split, chunk_size, chunk_overlap, separators):
    """
    Divide documentos en fragmentos utilizando separadores definidos.

    Parameters
    ----------
    docs_to_split : list of Document
        Documentos a fragmentar.
    chunk_size : int
        Tamaño máximo de cada fragmento en caracteres.
    chunk_overlap : int
        Número de caracteres que se solapan entre fragmentos.
    separators : list of str
        Lista de separadores para dividir el texto.

    Returns
    -------
    list of Document
        Lista de fragmentos generados a partir de los documentos originales.
    """
    if not docs_to_split:
        logger.debug("No hay documentos por dividir.")
        return []

    logger.info(
        f"Dividiendo {len(docs_to_split)} documento(s) en fragmentos..."
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap,
                                              separators=separators)

    chunks = splitter.split_documents(docs_to_split)

    logger.info(f"Generados {len(chunks)} fragmentos")
    return chunks
