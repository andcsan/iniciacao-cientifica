from ffnet import readdata, savenet, loadnet
from sklearn.cross_validation import KFold
import time

tempo = time.time()

data_dir = "../dados-treinamento/"
data_nome = "training-data.dat"
data = readdata(data_dir + data_nome, delimiter=" ")
rows, columns = data.shape

net = loadnet("../redes/default-net")

input = data[:, : columns - 1]
target = data[:, columns - 1]

kf = KFold(len(data), n_folds=4)

cont = 1
for train_index, test_index in kf:
    print("Treinando")
    net.train_tnc(input[train_index], target[train_index], maxfun=5000, messages=1)

    print("Testando")
    output, regression = net.test(input[test_index], target[test_index], iprint=2)

    print("Salvando rede KFold treinada")
    save_dir = "../redes/"
    save_nome = "cross-val-fold-" + str(cont) + "-trained-net"
    savenet(net, save_dir + save_nome)
    cont = cont + 1

print("Tempo de execução:", time.time() - tempo)
