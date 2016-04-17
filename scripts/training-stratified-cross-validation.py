from ffnet import readdata, savenet, loadnet
from numpy import genfromtxt
from sklearn import metrics
from math import sqrt, pow
import matplotlib.pyplot as plt
import time, sys, math


def roundBias(num, bias=0.5):
    """
    Arredonda um numero de acorda com um limiar definido
    """
    if num % 1 > bias:
        return math.ceil(num)
    else:
        return math.floor(num)


tempo = time.time()  # tempo inicial

netloc = sys.argv[1]  # local da rede padrão
dadosloc = sys.argv[2]  # local dos dados de treino
foldloc = sys.argv[3]  # local dos fold a serem abertos

dados = readdata(dadosloc, delimiter=" ")  # lendo dados de treinamento
rows, columns = dados.shape

input = dados[:, : columns - 1]  # dados de entrada
target = dados[:, columns - 1]  # dados de alvo

ind_treino = genfromtxt(foldloc, delimiter=" ", skip_footer=1, dtype="int")
ind_teste = genfromtxt(foldloc, delimiter=" ", skip_header=1, dtype="int")

TP, TN = 0, 0  # TruePositive e TrueNegative
FP, FN = 0, 0  # FalsePositive e FalseNegative

net = loadnet(netloc)  # carrega rede padrão

# treina a rede com os índices de treino
net.train_tnc(input[ind_treino], target[ind_treino], maxfun=5000, messages=1, nproc=6)

# teste a rede com os índices de teste
output, regression = net.test(input[ind_teste], target[ind_teste], iprint=2)

# cálculo de TruePositive FalseNegative
for alvo, saida in zip(target[ind_teste], output):
    saida = roundBias(saida, bias=0.5)
    if alvo == 1 and saida == 1:
        TP += 1
    elif alvo == 1 and saida == 0:
        FN += 1
    elif alvo == 0 and saida == 1:
        FP += 1
    elif alvo == 0 and saida == 0:
        TN += 1

print("TP:", TP, "TN:", TN)
print("FP:", FP, "FN:", FN)

print("Tempo de execução: ", time.time() - tempo)

# calculando curva roc
true_array = target[ind_teste]
score_array = output.ravel()

fpr, tpr, thresholds = metrics.roc_curve(true_array, score_array, pos_label=1)

# limiar ótimo distância euclidiana
deuc = [sqrt(pow(x - 0, 2) + pow(y - 1, 2)) for x, y in zip(fpr, tpr)]
print("Distância euclidiana", deuc)
print("Menor distância euclidiana: ", min(deuc))

# limiar ótimo distância Younden
dyoun = [y - x for x, y in zip(fpr, tpr)]
print("Distância Younden:", dyoun)
print("Maior distância Younden: ", max(dyoun))

# plotando a curva ROC
plt.plot([float(i) for i in fpr], [float(i) for i in tpr], "ro-")
plt.plot([0, 1], [0, 1])
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.show()
