from ffnet import ffnet, mlgraph, readdata, savenet, loadnet
import time

tempo = time.time()

data_dir = "../database/"
data_nome = "training-data.dat"
data = readdata(data_dir + data_nome, delimiter=" ")
rows, columns = data.shape

net = loadnet("../networks/default-net")

input = data[:, : columns - 1]
target = data[:, columns - 1]

print("Treinando")
net.train_tnc(input[:80], target[:80], maxfun=5000, messages=1)

print("Testando")
output, regression = net.test(input[80:], target[80:], iprint=2)

print("Salvando rede treinada")
save_dir = "../networks/"
save_nome = "holdout-trained-net"
savenet(net, save_dir + save_nome)

print("Tempo de execucao:", time.time() - tempo)
