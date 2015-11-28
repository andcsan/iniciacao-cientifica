from ffnet import ffnet, mlgraph, readdata, savenet, loadnet
from sklearn.cross_validation import StratifiedKFold
import time

tempo = time.time()

data_dir = "../database/"
data_nome = "training-data.dat"
data = readdata(data_dir + data_nome, delimiter=" ")
rows, columns = data.shape

net = loadnet("../networks/default-net")

input = data[:, : columns - 1]
target = data[:, columns - 1]

skf = StratifiedKFold(target, n_folds=4)

cont = 1
for train_index, test_index in skf:
    print("Treinando")
    net.train_tnc(input[train_index], target[train_index], maxfun=5000, messages=1)

    print("Testando")
    output, regression = net.test(input[test_index], target[test_index], iprint=2)

    print("Salvando rede StratifiedKFold treinada")
    save_dir = "../networks/"
    save_nome = "strat-cross-val-fold-" + repr(cont) + "-trained-net"
    savenet(net, save_dir + save_nome)
    cont = cont + 1

print("Tempo de execucao:", time.time() - tempo)
