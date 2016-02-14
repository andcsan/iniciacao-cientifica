from ffnet import readdata, savenet, loadnet
import time

tempo = time.time()

data_dir = "../dados-treinamento/"
data_nome = "training-data.dat"
data = readdata(data_dir + data_nome, delimiter=" ")
rows, columns = data.shape

net = loadnet("../redes/default-net")

input = data[:, : columns - 1]
target = data[:, columns - 1]

print("Treinando")
lim = len(input) - len(input) / 4
net.train_tnc(input[:lim], target[:lim], maxfun=5000, messages=1)

print("Testando")
output, regression = net.test(input[lim:], target[lim:], iprint=2)

print("Salvando rede treinada")
save_dir = "../redes/"
save_nome = "holdout-trained-net"
savenet(net, save_dir + save_nome)

print("Tempo de execução:", time.time() - tempo)
