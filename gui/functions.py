import numpy, os, random, cv2, ffnet


def create_database(classifier, color=False):
    posdir = classifier + "/pos/"
    negdir = classifier + "/neg/"
    save = classifier + "/database.dat"

    poslist = [posdir + i for i in os.listdir(posdir)]
    neglist = [negdir + i for i in os.listdir(negdir)]

    arq = open(save, mode="wb")

    all = poslist + neglist
    random.shuffle(all)

    for cur in all:
        if color:
            img = cv2.imread(cur)
            b = img[:, :, 0]
            g = img[:, :, 1]
            r = img[:, :, 2]
            img = numpy.concatenate((b, g, r))
        else:
            img = cv2.imread(cur, 0)
        img = img.ravel()

        if "neg" in cur:
            target = 0
            data = numpy.append(img, target)
        elif "pos" in cur:
            target = 1
            data = numpy.append(img, target)

        data = data.reshape((1, data.size))
        numpy.savetxt(arq, data, fmt="%d")


def create_cuts(image, savedir, size, offset):
    img = cv2.imread(image)
    img_h, img_w = img.shape

    os.chdir(savedir)

    name = "offset"
    ext = ".png"

    i, j = 0, 0
    for row in range(0, img_h - size, offset):
        for column in range(0, img_w - size, offset):
            window = img[row : row + size, column : column + size]
            cv2.imwrite(name + str(i) + "x" + str(j) + ext, window)
            j += 1
        i += 1


def create_dfnet(classifier, inp, h1, h2, out):
    save = classifier + "/default-net"

    connection = ffnet.mlgraph((inp, h1, h2, out))

    net = ffnet.ffnet(connection)
    ffnet.savenet(net, save)


def train_net(classifier):
    data = classifier + "/database.dat"
    dfnet = classifier + "/default-net"
    save = classifier + "/trained-net"

    data = ffnet.readdata(data, delimiter=" ")
    rows, columns = data.shape

    net = ffnet.loadnet(dfnet)
    net.randomweights()

    input = data[:, : columns - 1]
    target = data[:, columns - 1]

    print("Treinando")
    net.train_tnc(input[:], target[:], maxfun=5000, messages=1, nproc=6)

    print("Salvando rede treinada")
    ffnet.savenet(net, save)


def run(classifier, image, size, offset, color=False):
    os.chdir(classifier)

    net = classifier + "/trained-net"

    if color:
        img = cv2.imread(image)
    else:
        img = cv2.imread(image, 0)

    img_h, img_w = img.shape

    net = ffnet.loadnet(net)

    cells, cuts = 0, 0
    edge = []

    collision = False

    for row in range(0, img_h - size, offset):
        for column in range(0, img_w - size, offset):
            window = img[row : row + size, column : column + size]

            if color:
                b = window[:, :, 0]
                g = window[:, :, 1]
                r = window[:, :, 2]
                window = numpy.concatenate((b, g, r))

            window = numpy.ravel(window)

            target = [1]
            cuts += 1

            output = net.test([window], [target], iprint=0)

            if output >= 0.9:
                for a in edge:
                    if (
                        column < a[0] + 16
                        and column + size > a[0] - 16
                        and row < a[1] + 16
                        and row + size > a[1] - 16
                    ):
                        collision = True
                if not (collision):
                    cells += 1
                    edge.append([column + size / 2, row + size / 2])
                else:
                    collision = False
                    column += offset

    for a in edge:
        cv2.rectangle(
            img,
            (a[0] - size / 2, a[1] - size / 2),
            (a[0] + size / 2, a[1] + size / 2),
            (255, 0, 0),
            thickness=2,
        )
        cv2.circle(img, (a[0], a[1]), 32, (255, 0, 0), thickness=2)

    cv2.imwrite("imagem.png", img)
    cv2.destroyAllWindows()
