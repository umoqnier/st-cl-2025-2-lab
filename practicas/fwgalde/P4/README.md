# Práctica 4: Modelos del Lenguaje Neuronales
## Entorno de desarrollo
- Fedora Linux 40 (Workstation Edition)
- Python 3.11.11
- Jupyter
  - IPython          : 8.30.0
  - ipykernel        : 6.29.5
  - jupyter_client   : 8.6.3
  - jupyter_core     : 5.7.2
  - jupyter_server   : 2.15.0
  - jupyterlab       : 4.3.5
  - nbclient         : 0.10.2
  - nbconvert        : 7.16.6
  - nbformat         : 5.10.4
  - traitlets        : 5.14.3

## Dependencias
Para el correcto funcionamiento del programa se necesitan las siguientes dependencias:

- numpy=1.26.4
- nltk=3.9.1
- pytorch=2.4.1
- time
- random
- randint
- matplotlib=3.10.1
- scikit-learn=1.6.1
- umap-learn=0.5.7
- gensim=4.3.3
- tokenizers=0.21.1

Para más información se puede consultar el documento [requirements.yml](requirements.yml)

## Notas
- Se tuvieron que descargar lo siguientes elementos de nltk:

```
nltk.download('reuters')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

- Se utilizaron LLM's para la realización de documentación y en especial en la creación de funciones que me permitieran adaptar el modelo de subword tokenization; es decir, el extra.