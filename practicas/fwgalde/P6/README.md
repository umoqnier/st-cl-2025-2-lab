# Práctica 6: *Fine-tuning en producción*
## Entorno de desarrollo 🌐
- Ubuntu 22.04.4 LTS jammy 🐧
- Python 3.11.11 🐍
- Jupyter 📓
  - IPython          : 7.34.0 ⚙️
  - ipykernel        : 6.17.1 🚀
  - ipywidgets       : 8.1.5 🎛️
  - jupyter_client   : 8.6.3 📡
  - jupyter_core     : 5.7.2 🛠️
  - jupyter_server   : 2.12.5 🌐
  - jupyterlab       : 3.6.8 🖥️
  - nbclient         : 0.5.13 🤖
  - nbconvert        : 6.4.5 🔄
  - nbformat         : 5.10.4 📝
  - notebook         : 6.5.4 📒
  - qtconsole        : 5.6.1 💬
  - traitlets        : 5.7.1 🧩

## Descripción de la aplicación 💡

La aplicación web, desplegada en [Hugging Face Spaces](https://huggingface.co/spaces/fwgalde/dota2-toxic-detector-space), permite:

1. Ingresar un mensaje de chat de Dota 2.
2. Clasificar la toxicidad en tres niveles:

   * **non-toxic**
   * **mild toxicity**
   * **toxic**
3. Mostrar las probabilidades asociadas a cada categoría mediante una interfaz sencilla basada en Gradio.

## Dependencias 📦
Para el correcto funcionamiento del programa se necesitan las siguientes dependencias:

- numpy=1.26.4 🔢
- scikit-learn=1.2.2 🤖
- rich=14.0.0 🎨
- pandas=2.2.3 🐼
- transformers=4.51.1 🤗
- codecarbon=3.0.1 🌱
- huggingface-hub=0.30.2 ☁️

Para más información se puede consultar el documento [requirements.txt](requirements.txt) 📄


## Métricas de evaluación 📊
### Modelo propio 🐧

Se utilizaron 638 ejemplos del conjunto de prueba para evaluar el rendimiento:

|        Métrica |   Valor |
| -------------: | ------: |
| Pérdida (Loss) |  1.5071 |
|   **Accuracy** | 82.13 % |
|   **F1 Macro** | 76.30 % |

**Precisión por clase:**

* **Clase 0 (non-toxic)**: precision 0.96, recall 0.92, f1-score 0.94 (353 muestras)
* **Clase 1 (mild toxicity)**: precision 0.63, recall 0.63, f1-score 0.63 (118 muestras)
* **Clase 2 (toxic)**: precision 0.69, recall 0.76, f1-score 0.73 (167 muestras)

Estos resultados muestran un alto desempeño en la clase mayoritaria y un rendimiento aceptable en las clases minoritarias, con oportunidades de mejora mediante técnicas de balanceo de datos y preprocesamiento especializado.

---

*Generado con apoyo de herramientas de IA para documentación y formato.*

### Comparación con modelo baseline 🔍

El modelo fine-tuned original publicado por el creador del dataset usó **bert-base-uncased** y arrojó estas métricas sobre las mismas 638 muestras de prueba:

| Métrica        |  Valor |
| -------------- | -----: |
| Pérdida (Loss) | 0.9516 |
| Accuracy       | 79.78% |
| F1 Macro       | 73.52% |

**Precisión por clase (baseline):**

* **Clase 0 (non-toxic)**: precision 0.91, recall 0.91, f1-score 0.91 (353 muestras)
* **Clase 1 (mild toxicity)**: precision 0.60, recall 0.58, f1-score 0.59 (118 muestras)
* **Clase 2 (toxic)**: precision 0.70, recall 0.72, f1-score 0.71 (167 muestras)

Nuestra versión, basada en **roberta-base**, supera este baseline, alcanzando **82.13%** de accuracy y **76.30%** de F1 Macro, mejorando principalmente las métricas globales y el rendimiento en las clases minoritarias.

## Conclusiones 🏁

### ¿Resolvió la tarea? ¿Qué tan útil fue la app?

La aplicación logró cumplir el objetivo principal: detectar y clasificar la toxicidad en mensajes de chat de videojuegos (Dota 2) con un alto nivel de precisión y un F1 Macro competitivo. El flujo de trabajo de Gradio ofrece respuestas en tiempo real, facilitando su uso en moderación automática de comunidades o integración en bots de streaming.

### Retos y dificultades ⚠️

* **Selección de dataset:** Encontrar un conjunto de datos adecuado fue el mayor obstáculo. Aunque varios datasets estaban disponibles, muchos ejemplos eran ambiguos o no cubrían bien la jerga, lo que afectó la calidad de las predicciones.
* **Prueba de modelos preentrenados:** Se evaluaron múltiples backbones (DistilBERT, ALBERT), pero ninguno mejoró sustancialmente el desempeño inicial antes de llegar a RoBERTa.
* **Despliegue en producción:** El proceso de subir el modelo a Hugging Face requirió familiarizarse con la autenticación por tokens y con la estructura de repositorios (`config.json`, tokenizador, pesos), lo que generó varios errores de autorización y configuración.
* **Integración de CodeCarbon:** Las versiones de NVML de la GPU en entornos como Kaggle y Colab no soportaban las llamadas de energía total, por lo que fue necesario adaptar el tracker para medir solo CPU/RAM y garantizar un reporte sin fallos.
---

## Notas 📝
- Se utilizaron LLM’s 🤖 para la realización de documentación y formato del código.
- Toda la información sobre las emisiones de mi aplicación se puede encontrar en el archivo [emissions.csv](emissions.csv).
- La aplicación para Hugginface fue realizada con el SDK de Gradio y se puede ver el diseño de la "interfaz" en el archivo [app.py](app.py).