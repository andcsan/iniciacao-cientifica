import cv2, sys, os

imgloc = sys.argv[1]  # local da imagem a ser aberta
cortedir = sys.argv[2]  # diretório onde serão salvos os cortes

# abre a imagem no local definido
img = cv2.imread(imgloc, 1)

img_h, img_w, img_col = img.shape

# define o nome da imagem cortada, a extensão e o tamanho do corte
corte_nome = "blob"
corte_exts = ".png"
corte_tam = 256
corte_des = corte_tam / 2

# cria as pastas para armazenar as imagens cortadas
if not os.path.exists(cortedir):
    os.mkdir(cortedir)

os.chdir(cortedir)

# parâmetros para detector de blob
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
        cv2.imwrite(corte_nome + "_" + str(i) + corte_exts, corte_atual)
    else:
        print("Índice inexistente")
