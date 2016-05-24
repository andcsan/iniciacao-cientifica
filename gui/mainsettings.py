import Tkinter


class RedirectText(object):
    def __init__(self, text_wid):
        self.output = text_wid

    def write(self, string):
        self.output.insert(Tkinter.END, string)


title = "GUI - Célula / Micronúcleo"

# Warnings
train_warning = "A operação irá salvar uma rede treinada com o nome de \
trained-net no diretório escolhido. Deseja Continuar?"

dfnet_warning = "A operação irá salvar uma rede padrão com o nome de \
default-net no diretório escolhido. Deseja Continuar?"

database_warning = "A operação irá salvar um database com o nome de \
database.dat no diretório escolhido. Deseja Continuar?"
