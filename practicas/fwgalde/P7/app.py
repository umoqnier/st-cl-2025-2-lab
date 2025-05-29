import os
import shutil
import streamlit as st
import traceback

import config
import util_docs
import vector_db
import llm_services

from loguru import logger


logger.info("Iniciando app.py - carga de recursos globales...")
util_docs.ensure_dirs_exists([config.DOCS_PATH, config.PERSIST_DIR])

try:
    embedding_function = vector_db.load_embeddings(config.EMBEDDING_MODEL_NAME)
    logger.info("Función de embeddings cargada/obtenida de caché.")
except Exception as e:
    logger.exception("Fallo CRÍTICO al cargar modelo de embeddings.")
    st.error(
        f"Error fatal al cargar el modelo de embeddings: {e}. "
        "La aplicación no puede continuar."
    )
    st.stop()

try:
    db_instance = vector_db.load_vectordb(config.PERSIST_DIR,
                                          embedding_function)
    logger.info(
        "Instancia de VectorDB cargada/obtenida de caché. "
        f"Count: {vector_db.get_store_collection_count(db_instance)}")
except Exception as e:
    logger.exception("Fallo CRÍTICO al cargar VectorDB")
    st.error(
        f"Error fatal al cargar la base de datos vectorial: {e}. "
        "Intenta reiniciar el estudio."
    )
    db_instance = None

if "history" not in st.session_state:
    st.session_state.history = []
    logger.debug("Historial de chat inicializando en session_state")

if "needs_reset_triggered" not in st.session_state:
    st.session_state.needs_reset_triggered = False


if st.session_state.needs_reset_triggered:
    st.session_state.needs_reset_triggered = False
    logger.info("🚩 Iniciando proceso de reseteo...")
    try:
        if os.path.exists(config.DOCS_PATH):
            shutil.rmtree(config.DOCS_PATH, ignore_errors=True)
        if os.path.exists(config.PERSIST_DIR):
            shutil.rmtree(config.PERSIST_DIR, ignore_errors=True)

        util_docs.ensure_dirs_exists([config.DOCS_PATH, config.PERSIST_DIR])
        logger.info("Directorios DOCS_PATH y PERSIST_DIR recreados.")

        st.session_state.history = []
        logger.info("Historial de chat reseteado.")

        vector_db.load_embeddings.clear()
        vector_db.load_vectordb.clear()
        llm_services.load_llm.clear()
        logger.info("Cachés de recursos @st.cache_resource limpiados.")

        reset_feedback_msg = ("¡Estudio reseteado! "
                              "La aplicación se ha actualizado. "
                              "Reiniciela para correcto funcionamiento (bugs)")

        embedding_function = vector_db.load_embeddings(
            config.EMBEDDING_MODEL_NAME
        )

        db_instance = vector_db.load_vectordb(config.PERSIST_DIR,
                                              embedding_function)
        logger.info("Instancia de DB recargada después del reset.")

    except Exception as e:
        logger.exception("Error durante el proceso de reseteo.")
        reset_feedback_msg = f"Error durante el reset: {e}"

    if "Error" in reset_feedback_msg:
        st.sidebar.error(reset_feedback_msg)
    else:
        st.sidebar.success(reset_feedback_msg)



st.sidebar.header("📂 Gestión de documentos")

uploaded_files_ui = st.sidebar.file_uploader(
    "Sube .txt/.pdf", accept_multiple_files=True, type=["txt", "pdf"]
)


if st.sidebar.button("🔄 Indexar nuevos docs"):
    logger.info(
        "Botón 'Indexar archivos subidos' presionado"
        f"{len(uploaded_files_ui) if uploaded_files_ui else 0} archivos."
    )

    if not uploaded_files_ui:
        st.sidebar.warning("Por favor, sube al menos un archivo.")
        logger.warning("Intento de indexación sin archivos subidos.")

    elif db_instance is None:
        st.sidebar.error(
            "La base de datos vectorial no está disponible. Revisa errores."
        )
        logger.error("Intento de indexación pero db_instance es None.")

    else:
        logger.debug("Guardando archivos subidos...")
        save_paths, feedback_save = util_docs.save_uploaded_files(
            uploaded_files_ui,
            config.DOCS_PATH
        )

        for msg in feedback_save:
            if "❌" in msg: st.sidebar.error(msg)
            else: st.sidebar.info(msg)


        if save_paths:
            logger.debug("Cargando documentos desde rutas guardadas")

            docs_to_process, feedback_load =\
                util_docs.load_docs_from_files_paths(save_paths)

            for msg in feedback_load:
                if "❌" in msg: st.sidebar.error(msg)
                else: st.sidebar.info(msg)

            if docs_to_process:
                logger.debug("Dividiendo documentos en chunks...")
                chunks = util_docs.split_langchain_docs(docs_to_process,
                                                        config.CHUNK_SIZE,
                                                        config.CHUNK_OVERLAP,
                                                        config.SEPARATORS)
                if chunks:
                    logger.debug("Añadiendo chunks a VectorDB...")
                    st.sidebar.text(f"Procesando {len(chunks)} fragmentos...")

                    count_added, error_add = vector_db.add_documents_to_store(
                        db_instance,
                        chunks
                    )

                    if error_add:
                        st.sidebar.error(error_add)
                    elif count_added > 0:
                        st.sidebar.success(
                            f"✅ ¡Indexados {count_added} nuevos fragmentos!"
                        )
                        st.sidebar.info(
                            "Total en DB "
                            f"{vector_db.get_store_collection_count(db_instance)}"
                            " nuevos fragmentos!"
                        )
                    else:
                        st.sidebar.info(
                            "No se añadieron nuevos fragmentos "
                            "(podrían ser duplicados o vacíos)"
                        )
                else:
                    st.sidebar.warning(
                        f"⚠️ No hay archivos nuevos para procesar"
                    )
            elif save_paths:
                st.sidebar.warning("No se pudo extraer contenido válido "
                                   "de los archivos guardados.")
        elif uploaded_files_ui:
            st.sidebar.error("No se pudieron guardar los archivos subidos "
                             "para procesarlos")

if st.sidebar.button("🗑️ Reiniciar estudio"):
    logger.info("Botón 'Reiniciar estudio' presionado.")
    st.session_state.needs_reset_triggered = True
    st.rerun()

st.title("🧠 StudyBuddy")
st.markdown("Chat de estudio sobre tus documentos locales.")


k_slider_value = st.sidebar.slider("Top k (fragmentos)", 1, 10, 3,
                                   key="top_k_chat_slider")

chat_is_ready = False
qa_chain = None

if db_instance is not None:
    try:
        llm_instance = llm_services.load_llm(config.LLM_MODEL,
                                             config.LLM_TEMPERATURE,
                                             config.LLM_NUM_THREAD)
        logger.info("Instancia de LLM cargada/obtenida de caché.")

        retriever = vector_db.get_retriever(db_instance, k_slider_value)

        if retriever:
            qa_chain = llm_services.create_qa_chain(llm_instance, retriever)
            if qa_chain:
                chat_is_ready = True
                logger.info("Cadena QA creada y lista para el chat.")

            else:
                st.error("No se pudo crear la cadena de QA")
                logger.error("Fallo al crear qa_chain")
        else:
            st.error("No se pudo obtener el retriever de la base de datos.")
            logger.error("Fallo al obtener el retriever.")

    except Exception as e:
        log.exception("Error al configurar los componentes principales "
                      "de la app (LLM, QA chain)")
        st.error(f"Error al configurar la cadena QA: {e}")

else:
    st.error("Base de datos vetorial no disponible. El chat está deshabilitado")
    logger.warning("Chat deshabilitado porque db_instance es None")


if chat_is_ready:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query_ui = st.chat_input("Escribe tu pregunta sobre los documentos:")
    if user_query_ui:
        logger.info(f"Usuario pregunó '{user_query_ui[:22]}...'")
        st.session_state.history.append({"role": "user",
                                         "content": user_query_ui})

        with st.chat_message("user"):
            st.markdown(user_query_ui)

        with st.spinner("🤔 StudyBuddy está pensando..."):
            answer = ""

            try:
                if vector_db.get_store_collection_count(db_instance) == 0:
                    answer = ("La base de conocimiento está vacía."
                                  " Por favor, sube e indexa documentos")

                    logger.warning("Intento de consulta con DB vacía.")

                elif qa_chain:
                    output = qa_chain.invoke({"query": user_query_ui})
                    answer = output["result"]

                    st.session_state.history.append({"role":"assistant",
                                                     "content":answer})

                    logger.info(f"Respuesta generada {answer[:22]}...")

                else:
                    answer = "Error: La cadena de QA no está disponible."
                    logger.error(
                        "qa_chain es None dentro del bloque de chat listo"
                    )

            except Exception as e:
                logger.exception("Error durante la invocación de la cadena QA.")
                answer = (
                    f"Ocurrio un problema al procesar tu pregunta: {e}"
                )

            with st.chat_message("assistant"):
                st.markdown(answer)

else:
    st.warning("El chat no está disponible debido a errores de "
               "incialización o porque la DB está vacía")

logger.info("🏁 Ejecución de app.py completada (o rerun)")
