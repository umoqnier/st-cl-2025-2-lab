$ cat README.md

# **Práctica 5**

## **Comentarios**

En este folder se encuentra la práctica 5 y solo es necesario
correr cada celda en orden cronológico. El código fue hecho en un
notebook de Colab. Muchas de las funciones
fueron tomadas de la práctica vista en clase. 

Además para correr el notebook se necesita el modelo preentrenado
en clase que se llama pos_tagger_rnn_cpu_9.pth. Dado que es muy grande
lo guardé en la nube y se puede descargar de la siguiente liga:
https://drive.google.com/file/d/1C62yor0qrIiV0T0Gw4BIYSNuUOnxNGP4/view?usp=sharing

Para correr la libreta se necesita que este archivo se encuentre
en la carpeta /content/. También es necesario cargar los demás
archivos a la misma carpeta que son diccionarios que relacionan
cada índice con la palabra o POS correspondiente. Todos se encuentran
en la carpeta y se llaman:

idx2pos
idx2word
word2idx
pos2idx


Solo hace falta correr el código para cargarlos al notebook

Dependencias Utilizadas

pickle
torch
nltk
sklearn.decomposition
sklearn.cluster
matplotlib.pyplot
string
random
numpy