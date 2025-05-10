# Práctica de Procesamiento de Lenguaje Natural

## Descripción
Esta práctica se centró en entender los modelos neuronales del lenguaje, algunas de sus ventajas y su lugar en el proceso del marco de trabaajo de PLN. Se utilizaron diversas bibliotecas de Python para llevar a cabo las actividades propuestas.

## Bibliotecas Utilizadas
- NLTK
- Numpy
- Pandas
- Matplotlib
- Scikit-learn
- Torch

## Actividades
1. **Generación de vectores de palabras (embedings) usando la arquitectura de Benjio**: 
    - Se generaron vectores de palabras a partir del entrenamiento de un modelo de lenguaje utilizando la arquitectura de Benjio.
    - Se utilizó la biblioteca `torch` para implementar el modelo y generar los vectores de palabras.
    - Se entrenó el modelo utilizando un corpus de texto y se usó PCA para reducir la dimensionalidad de los vectores generados.
    - Se graficaron los vectores generados para visualizar la distribución de las palabras en el espacio vectorial.
    
2. **Verificación de la calidad de la representación**:
    - Se utilizó la similitud coseno para verificar la calidad de la representación de los vectores generados.
    - Se calcularon las similitudes entre pares de palabras y se compararon con las similitudes esperadas.
    - Se ejemplificó la calidad de la representación utilizando analogías de palabras, como "rey - hombre + mujer = reina".

## Requisitos
- Python 3.10
- Las bibliotecas mencionadas anteriormente deben estar instaladas.

## Instalación
Para instalar las bibliotecas necesarias, puede utilizar el siguiente comando:
```bash
pip install nltk numpy pandas matplotlib scikit-learn torch
```

## Ejecución
Para ejecutar los scripts de la práctica, asegúrese de tener todas las bibliotecas instaladas y ejecute los archivos correspondientes desde su entorno de desarrollo.

## Contribuciones
Las contribuciones a este proyecto son bienvenidas. Por favor, cree un fork del repositorio y envíe un pull request con sus mejoras.