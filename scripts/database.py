from random import shuffle
from cv2 import imread

import numpy as np
import os, sys

posdir = sys.argv[1]  # diretório das imagens positivas
negdir = sys.argv[2]  # diretório das imagens negativas
saveloc = sys.argv[3]  # local para salvar o arquivo de dados

nomespos = os.listdir(posdir)  # lista de nomes de imagens positivas
nomesneg = os.listdir(negdir)  # lista de nomes de imagens negativas

# lista de diretórios de cada imagem positiva
listapos = [posdir + x for x in nomespos]

# lista de diretórios de cada imagem negativa
listaneg = [negdir + x for x in nomesneg]

arq = open(saveloc, mode="wb")

alldir = listapos + listaneg  # lista com todos os diretórios de cada imagem
shuffle(alldir)  # embaralha a lista

for atualdir in alldir:
    img = imread(atualdir, 0)  # carrega a imagem em grayscale

    if "neg" in atualdir:
        target = 0
        dado = np.append(img.ravel(), target)
    elif "pos" in atualdir:
        target = 1
        dado = np.append(img.ravel(), target)

    dado = dado.reshape((1, dado.size))
    np.savetxt(arq, dado, fmt="%d")
