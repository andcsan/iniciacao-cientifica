import cv2, sys, os

# local da imagem a ser aberta
imgloc = sys.argv[1]
cortedir = sys.argv[2]
corte_tam = int(sys.argv[3])
offset = int(sys.argv[4])

# abre a imagem no local definido e coleta suas dimensões
img = cv2.imread(imgloc, 1)
img_h, img_w, img_col = img.shape

# define o nome da imagem cortada, a extensão e o tamanho do corte
corte_nome = "offset"
corte_exts = ".png"

# cria as pastas para armazenar as imagens cortadas
if not os.path.exists(cortedir):
    os.mkdir(cortedir)

os.chdir(cortedir)

# laços de corte
i, j = 0, 0
for linha in range(0, img_h - corte_tam, offset):
    for coluna in range(0, img_w - corte_tam, offset):

        corte_atual = img[linha : linha + corte_tam, coluna : coluna + corte_tam]

        if len(corte_atual) and len(corte_atual[0]) == corte_tam:
            cv2.imwrite(
                corte_nome + "_" + str(i) + "_" + str(j) + corte_exts, corte_atual
            )
        else:
            print("Imagem fora de padrão", corte_tam)

        j += 1
    i += 1
    j = 0
