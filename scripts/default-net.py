from ffnet import ffnet, mlgraph, savenet
import sys, time

tempo = time.time()

_in_ = int(sys.argv[1])  # quantidade de entradas da rede
_hidden1_ = int(sys.argv[2])  # quantidade na primeira hidden layer
_hidden2_ = int(sys.argv[3])  # quantidade na segunda hidden layer
_out_ = int(sys.argv[4])  # quantidade de saídas

connection = mlgraph((_in_, _hidden1_, _hidden2_, _out_))
net = ffnet(connection)
savenet(net, "../redes/default-net")

print("Tempo de execução", time.time() - tempo)
