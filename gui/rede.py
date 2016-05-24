from sklearn.cross_validation import StratifiedKFold
import ffnet, settings


class Rede(object):
    def __init__(self):
        self.imsize = settings.imgsize
        self.offset = settings.offset

        self.inp = settings.inp
        self.h1 = settings.h1
        self.h2 = settings.h2
        self.out = settings.out
        self.net = ffnet.loadnet(settings.dfnetloc)

        self.nfolds = settings.nfolds
        self.folds = []

        self.data = ffnet.readdata(settings.dataloc, delimiter=" ")
        self.rows, self.columns = self.data.shape
        self.input = self.data[:, : self.columns - 1]
        self.target = self.data[:, self.columns - 1]

        self.maxfun = settings.maxfun

        def training_all(self):
            self.net.train_tnc(self.input[:], self.target[:], self.maxfun, messages=1)

        def training_cvs(self):
            skf = StratifiedKFold(y=self.target, n_folds=self.nfolds)
            for ind_treino, ind_teste in skf:
                self.folds.append([ind_treino, ind_teste])
