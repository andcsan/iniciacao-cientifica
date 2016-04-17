from sklearn.cross_validation import StratifiedKFold
import numpy as np
import sys

dadosloc = sys.argv[1]
savedir = sys.argv[2]
nfolds = sys.argv[3]

dados = np.loadtxt(dadosloc, delimiter=" ")
rows, columns = dados.shape

target = dados[:, columns - 1]

skf = StratifiedKFold(y=target, n_folds=int(nfolds))

x = 1

for ind_treino, ind_teste in skf:
    arq = open(savedir + "fold" + repr(x) + ".dat", mode="wb")

    dado = ind_treino.reshape((1, ind_treino.size))
    np.savetxt(arq, dado, fmt="%d")

    dado = ind_teste.reshape((1, ind_teste.size))
    np.savetxt(arq, dado, fmt="%d")
    x += 1
