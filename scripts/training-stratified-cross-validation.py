from ffnet import readdata, savenet, loadnet
from numpy import genfromtxt

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
saveloc = sys.argv[4]  # local para salvar os treinamentos

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
net.train_tnc(input[ind_treino], target[ind_treino], maxfun=1000, messages=1, nproc=4)

# teste a rede com os índices de teste
output, regression = net.test(input[ind_teste], target[ind_teste], iprint=2)

# cálculo de TruePositive FalseNegative
for alvo, saida in zip(target[ind_teste], output):
    saida = roundBias(saida, bias=0.8)
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

savenet(net, saveloc)  # salva a rede treinada com o fold

print("Tempo de execução: ", time.time() - tempo)
