
import gradio as gr
from StudyBuddy import StudyBuddyAssistant
import os
from pathlib import Path
import shutil

DEFAULT_DOCS_DIR = "./pdfs_notes"
DEFAULT_PERSIST_DIR = "./documents_vector_db"

def initialize_assistant(model_name, embedding_model, documents_dir, persist_dir):
    """Inicializa el asistente de estudio con los parámetros proporcionados por el usuario"""
    try:
        Path(documents_dir).mkdir(parents=True, exist_ok=True)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        assistant = StudyBuddyAssistant(
            model_name=model_name,
            embedding_model=embedding_model,
            documents_dir=documents_dir,
            persist_directory=persist_dir
        )
        return assistant, "✅ Asistente inicializado correctamente"
    except Exception as e:
        return None, f"❌ Error al inicializar: {str(e)}"

def ask_question(assistant, question, history):
    if assistant is None:
        return history + [[question, "⚠️ Por favor inicializa el asistente primero"]]

    try:
        response = assistant.ask(question, stream=False)
        return history + [[question, response]]
    except Exception as e:
        return history + [[question, f"❌ Error: {str(e)}"]]

def upload_files(assistant, files):
    if not files:
        return "⚠️ No se seleccionaron archivos"
    if assistant is None:
        return "⚠️ Inicializa el asistente primero"

    try:
        saved_files = []
        for file in files:
            dest = os.path.join(assistant.documents_dir, os.path.basename(file.name))
            shutil.copy(file.name, dest)
            saved_files.append(dest)

        assistant.add_documents(saved_files)
        return f"✅ {len(saved_files)} archivos añadidos correctamente"
    except Exception as e:
        return f"❌ Error al subir archivos: {str(e)}"

def create_ui():
    with gr.Blocks(title="StudyBuddy - Asistente de Estudio", theme="soft") as demo:
        assistant_state = gr.State()

        gr.Markdown("""
        # 🎓 StudyBuddy - Asistente de Estudio Inteligente
        Transforma tus apuntes en material de aprendizaje efectivo
        """)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("⚙️ Configuración del Asistente", open=True):
                    model_name = gr.Dropdown(["llama3.2:3b"], label="Modelo LLM", value="llama3.2:3b")
                    embedding_model = gr.Dropdown(["nomic-embed-text:latest", "bge-m3:latest"], label="Modelo de Embeddings", value="nomic-embed-text:latest")
                    documents_dir = gr.Textbox(label="Directorio de Documentos", value=DEFAULT_DOCS_DIR)
                    persist_dir = gr.Textbox(label="Directorio de Base de Datos", value=DEFAULT_PERSIST_DIR)
                    init_btn = gr.Button("Inicializar Asistente")
                    init_status = gr.Markdown()

                with gr.Accordion("📤 Subir Archivos", open=True):
                    file_upload = gr.File(file_count="multiple", file_types=[".pdf"])
                    upload_btn = gr.Button("Procesar Archivos")
                    upload_status = gr.Markdown()

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=("assets/user.png", "assets/bot.png"))
                msg = gr.Textbox(placeholder="Escribe tu pregunta sobre el material de estudio...", label="Pregunta")
                submit_btn = gr.Button("Enviar")
                clear_btn = gr.ClearButton([msg, chatbot], value="Limpiar")

        init_btn.click(initialize_assistant, inputs=[model_name, embedding_model, documents_dir, persist_dir], outputs=[assistant_state, init_status])
        upload_btn.click(upload_files, inputs=[assistant_state, file_upload], outputs=upload_status)
        msg.submit(ask_question, inputs=[assistant_state, msg, chatbot], outputs=[chatbot])
        submit_btn.click(ask_question, inputs=[assistant_state, msg, chatbot], outputs=[chatbot])

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
