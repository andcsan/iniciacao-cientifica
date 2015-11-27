import cv2
import numpy
import sys
import os

# local da imagem a ser aberta
img_dir = "../images/grouped/"
img_nome = sys.argv[1]
img_dir = img_dir + img_nome

# abre a imagem no local definido e coleta suas dimensoes
img = cv2.imread(img_dir, 1)
img_h, img_w, img_col = img.shape

# define o nome da imagem cortada, a extensao e o tamanho do corte
corte_nome = "offset"
corte_exts = ".png"
corte_tam = 256

# define o deslocamento entre cada corte
offset = 64

# cria as pastas para armazenar as imagens cortadas
corte_dir = "../images/cuts/offset/"
if not os.path.exists(corte_dir):
    os.mkdir(corte_dir)

pasta = corte_dir + img_nome[: len(img_nome) - 4]
if not os.path.exists(pasta):
    os.mkdir(pasta)

os.chdir(pasta)

# lacos de corte
i, j = 0, 0
for linha in range(0, img_h - corte_tam, offset):
    for coluna in range(0, img_w - corte_tam, offset):

        corte_atual = img[linha : linha + corte_tam, coluna : coluna + corte_tam]

        if len(corte_atual) and len(corte_atual[0]) == corte_tam:
            cv2.imwrite(
                corte_nome + "_" + repr(i) + "_" + repr(j) + corte_exts, corte_atual
            )
        else:
            print("Imagem fora de padrao", corte_tam)

        j += 1
    i += 1
    j = 0
