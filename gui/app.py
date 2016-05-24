from Tkinter import *
import tkFileDialog as tkf
import tkMessageBox as tkm
import tkSimpleDialog as tks
import mainsettings as ms
import functions as fn
import os


class mainWindow(object):
    def __init__(self, rootmaster):
        self.initUI(rootmaster)

    def seldir_event(self):
        self.classifier_dir = tkf.askdirectory().encode("utf-8")
        if self.classifier_dir:
            if os.path.exists(self.classifier_dir + "/pos/") and os.path.exists(
                self.classifier_dir + "/neg/"
            ):
                self.info_label["text"] = self.classifier_dir
            else:
                tkm.showwarning(message="Classificador inválido")
        else:
            self.classifier_dir = []

    def cut_event(self):
        imgtype = [("Imagens", ".png .jpg .jpeg .JPG .PNG")]
        image = tkf.askopenfilename(title="Imagem", filetypes=imgtype).encode("utf-8")
        if image:
            save = tkf.askdirectory(title="Pasta").encode("utf-8")
            if save:
                size = tks.askinteger("Tamanho", "Tamanho de recorte")
                offset = tks.askinteger("Deslocamento", "Tamanho de deslocamento")
                fn.create_cuts(image, save, size, offset)

    def database_event(self):
        if tkm.askyesno(title="Atenção", message=ms.database_warning):
            if self.classifier_dir:
                if tkm.askyesno(message="Colorido?"):
                    fn.create_database(self.classifier_dir, color=True)
                else:
                    fn.create_database(self.classifier_dir)
            else:
                tkm.showwarning(message="Classificador inválido")

    def dfnet_event(self):
        if tkm.askyesno(title="Atenção", message=ms.dfnet_warning):
            if self.classifier_dir:
                inp = tks.askinteger("Input", "Entrada")
                h1 = tks.askinteger("Hidden", "Hidden1")
                h2 = tks.askinteger("Hidden", "Hidden2")
                out = tks.askinteger("Out", "Saídas")
                fn.create_dfnet(self.classifier_dir, inp, h1, h2, out)
            else:
                tkm.showwarning(message="Classificador inválido")

    def train_event(self):
        if tkm.askyesno(title="Atenção", message=ms.train_warning):
            if self.classifier_dir:
                fn.train_net(self.classifier_dir)
            else:
                tkm.showwarning(message="Classificador inválido")

    def run_event(self):
        imgtype = [("Imagens", ".png .jpg .jpeg .JPG .PNG")]
        image = tkf.askopenfilename(title="Imagem", filetypes=imgtype).encode("utf-8")
        if image:
            size = tks.askinteger("Tamanho", "Tamanho de recorte")
            offset = tks.askinteger("Deslocamento", "Tamanho de deslocamento")
            if tkm.askyesno(message="Colorido?"):
                fn.run(self.classifier_dir, image, size, offset, color=True)
            else:
                fn.run(self.classifier_dir, image, size, offset)

    def initUI(self, rootmaster):
        # main frame
        self.button_size = 20
        self.main_frame = Frame(rootmaster)
        self.main_frame.pack(side="bottom", fill="both")

        # buttons
        # diretório
        self.dir_button = Button(
            self.main_frame, text="Selecionar classificador", command=self.seldir_event
        )
        self.dir_button.configure(width=self.button_size)
        self.dir_button.pack(side="left")

        # recorte
        self.cut_button = Button(
            self.main_frame, text="Recortar imagem", command=self.cut_event
        )
        self.cut_button.configure(width=self.button_size)
        self.cut_button.pack(side="left")

        # database
        self.database_button = Button(
            master=self.main_frame, text="Criar database", command=self.database_event
        )
        self.database_button.configure(width=self.button_size)
        self.database_button.pack(side="left")

        # criar rede
        self.dfnet_button = Button(
            master=self.main_frame, text="Criar rede", command=self.dfnet_event
        )
        self.dfnet_button.configure(width=self.button_size)
        self.dfnet_button.pack(side="left")

        # treinar rede
        self.train_button = Button(
            master=self.main_frame, text="Treinar rede", command=self.train_event
        )
        self.train_button.configure(width=self.button_size)
        self.train_button.pack(side="left")

        # rodar rede neural para detectar células
        self.run_button = Button(
            master=self.main_frame, text="Detectar células", command=self.run_event
        )
        self.run_button.configure(width=self.button_size)
        self.run_button.pack(side="left")

        # frame
        self.text_frame = Frame(rootmaster)
        self.text_frame.pack(side="bottom", expand=True, fill="both")

        # texto
        self.out_text = Text(master=self.text_frame)
        self.out_text.pack(side="left", expand=True, fill="both")

        self.scrollbar = Scrollbar(master=self.text_frame)
        self.scrollbar.pack(side="left", fill="y")

        self.out_text.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.out_text.yview)
        self.info_label = Label(rootmaster, text="<Diretorio vazio>")
        self.info_label.pack(side="left")

        self.classifier_dir = []


root = Tk()
root.title(ms.title)
mainWindow(root)
root.mainloop()
