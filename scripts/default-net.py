from ffnet import ffnet, mlgraph, savenet
import sys, time

tempo = time.time()

_in_ = int(sys.argv[1])
_hidden1_ = int(sys.argv[2])
_hidden2_ = int(sys.argv[3])
_out_ = int(sys.argv[4])

connection = mlgraph((_in_, _hidden1_, _hidden2_, _out_))
net = ffnet(connection)
savenet(net, "../networks/default-net")

print("Tempo de execucao", time.time() - tempo)
