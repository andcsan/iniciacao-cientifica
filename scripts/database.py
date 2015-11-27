import numpy
import cv2
import os
import sys

# img_dir = '../banco-imagens/recortes/mix/'
# img_dir = '../banco-imagens/teste/'

img_dir = sys.argv[1]

print("Carregando lista de imagens")
img_lista = os.listdir(img_dir)

print("Criando banco de treinamento")
treino_dir = "../database/"
treino_data = open(treino_dir + "training-data.dat", "wb")

print("Inserindo targets correspondentes")
for i in range(0, len(img_lista), 1):

    dir_atual = img_dir + img_lista[i]
    img_atual = cv2.imread(dir_atual, 0)

    if img_lista[i][0] == "b":
        target = 1
        dado_atual = numpy.append(img_atual.ravel(), target)
    elif img_lista[i][0] == "o":
        target = 0
        dado_atual = numpy.append(img_atual.ravel(), target)

    print(
        repr(i + 1)
        + " Imagem: "
        + img_lista[i]
        + " .................... Target: "
        + repr(target)
    )

    dado_atual = dado_atual.reshape(1, len(dado_atual))
    numpy.savetxt(treino_data, dado_atual, fmt="%d")
