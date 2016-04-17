from ffnet import loadnet
import cv2, numpy, sys, time

tempo = time.time()

img_dir = sys.argv[1]  # imagem a ser aberta
rede_dir = sys.argv[2]  # rede a ser carregada
corte_tam = int(sys.argv[3])
offset = int(sys.argv[4])

# abre a imagem no local definido e coleta suas dimensões
img = cv2.imread(img_dir, 0)
img_h, img_w = img.shape

net = loadnet(rede_dir)

cell_counter = 0
cut_counter = 0
edge = []

linha = 0
coluna = 0
colision = False

while linha < img_h - corte_tam:
    while coluna < img_w - corte_tam:
        sample = img[linha : linha + corte_tam, coluna : coluna + corte_tam]
        cut_counter += 1

        if len(sample) and len(sample[0]) == corte_tam:
            aux = sample
            sample = numpy.append(sample.ravel(), 1)
            sample = numpy.array([sample[: len(sample) - 1]])
            target = numpy.array([[1]])
            output, regression = net.test(sample, target, iprint=0)
            if output >= 0.8:
                for a in edge:
                    if (
                        coluna < a[0] + 32
                        and coluna + corte_tam > a[0] - 32
                        and linha < a[1] + 32
                        and linha + corte_tam > a[1] - 32
                    ):
                        colision = True
                if not (colision):
                    cell_counter += 1
                    print("Célula identificada", output)
                    edge.append([coluna + corte_tam / 2, linha + corte_tam / 2])
                else:
                    cut_counter -= 1
                    coluna += offset
                    colision = False
        else:
            print("Imagem fora de padrão", corte_tam)
        print(cell_counter)
        coluna += offset
    coluna = 0
    linha += offset

for a in edge:
    cv2.rectangle(
        img,
        (a[0] - corte_tam / 2, a[1] - corte_tam / 2),
        (a[0] + corte_tam / 2, a[1] + corte_tam / 2),
        (255, 0, 0),
        thickness=3,
    )

cv2.imwrite("imagem.png", img)
cv2.destroyAllWindows()

print(cell_counter)
print(cut_counter)
print("Tempo de execução:", time.time() - tempo)
