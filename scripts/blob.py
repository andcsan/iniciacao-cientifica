import cv2
import numpy
import sys
import os

# local da imagem a ser aberta
img_dir = "../images/grouped/"
img_nome = sys.argv[1]
img_dir = img_dir + img_nome

# abre a imagem no local definido
img = cv2.imread(img_dir, 1)
img_h, img_w, img_col = img.shape

# define o nome da imagem cortada, a extensao e o tamanho do corte
corte_nome = "blob"
corte_exts = ".png"
corte_tam = 256
corte_des = corte_tam / 2

# cria as pastas para armazenar as imagens cortadas
corte_dir = "../images/cuts/blob/"
if not os.path.exists(corte_dir):
    os.mkdir(corte_dir)

pasta = corte_dir + img_nome[: len(img_nome) - 4]
if not os.path.exists(pasta):
    os.mkdir(pasta)

os.chdir(pasta)

# parametros para detector de blob
parametros = cv2.SimpleBlobDetector_Params()

parametros.minThreshold = 5
parametros.maxThreshold = 80

parametros.filterByArea = True
parametros.minArea = 30

parametros.filterByCircularity = True
parametros.minCircularity = 0.1

parametros.filterByConvexity = True
parametros.minConvexity = 0.1

parametros.filterByInertia = True
parametros.minInertiaRatio = 0.1

detector = cv2.SimpleBlobDetector(parametros)
keypoints = detector.detect(img)

for i in range(0, len(keypoints), 1):
    x, y = keypoints[i].pt

    iy, fy = y - corte_des, y + corte_des
    ix, fx = x - corte_des, x + corte_des

    if iy > 0 and ix > 0 and fy < img_h and fx < img_w:
        corte_atual = img[iy:fy, ix:fx]
        cv2.imwrite(corte_nome + "_" + repr(i) + corte_exts, corte_atual)
    else:
        print("Indice inexistente")
