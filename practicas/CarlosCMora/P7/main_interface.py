from StudyBuddy import StudyBuddyAssistant
import streamlit as st
import os

def main():
    st.title("📚 StudyBuddy Assistant")
    st.subheader("Tu asistente de estudio personalizado")

    # Sidebar para la configuración
    with st.sidebar:
        st.header("Configuración")
        model_name = st.selectbox("Modelo LLM:", ["llama3.2"]) 
        embedding_model = st.selectbox("Modelo de Embeddings:", ["nomic-embed-text:latest", "all-MiniLM-L6-v2"]) 
        documents_dir = "pdfs_notes" 
        # Asegurarse de que el directorio de documentos exista
        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir)
            st.info(f"Se ha creado el directorio '{documents_dir}'. Por favor, carga tus archivos PDF allí.")

        uploaded_files = st.file_uploader("Carga tus documentos PDF aquí:", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(documents_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Se han guardado {len(uploaded_files)} documentos en '{documents_dir}'.")

        st.info("Los documentos cargados se procesarán al realizar la primera pregunta.")

    # Inicializar el asistente (se inicializará al hacer la primera pregunta)
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = None

    # Área de chat
    st.header("Chat")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! Soy tu asistente para estudiar. ¿Cómo te gustaría que te ayude hoy?"}]

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if st.session_state["assistant"] is None:
                st.session_state["assistant"] = StudyBuddyAssistant(
                    model_name=model_name,
                    embedding_model=embedding_model,
                    documents_dir=documents_dir
                )

            for chunk in st.session_state["assistant"].ask(prompt, stream=True):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state["messages"].append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()