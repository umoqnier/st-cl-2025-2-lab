import subprocess
import streamlit as st
from langchain_ollama import OllamaLLM

# 1) Lista modelos instalados
res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
disponibles = res.stdout.strip()

# 2) Lee el modelo deseado (o pon “gemma3:4b” directamente)
model = st.sidebar.text_input("Modelo Ollama", value="gemma3:4b")
temp  = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.2)

# 3) Intenta instanciar el LLM y atrapa el error
try:
    llm = OllamaLLM(model=model, temperature=temp)
except ValueError as e:
    st.sidebar.error(f"❌ Error cargando '{model}': {e}")
    st.sidebar.markdown("**Modelos disponibles:**\n```\n" + disponibles + "\n```")
    st.stop()
