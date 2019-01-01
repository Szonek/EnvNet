from tkinter import *
import numpy as np

class Gui:
    """
    This class defines the simple GUI to present NameNet model.
    """
    def __init__(self, network):
        self.network = network
        self.master = Tk()
        self.__configure_window()
        self.last_name = Entry(self.master)
        self.last_name.place(x=200, y=50)
        Label(self.master, text="Last name:", font="Verdana 10").place(x=100, y=50)
        Button(self.master, text="Predict", command=self.__predict,width=15,height=4).place(x=100, y=100)
        self.result_0 = Label(self.master, text="0:", font="Verdana 10")
        self.result_1 = Label(self.master, text="1:", font="Verdana 10")
        self.result_2 = Label(self.master, text="2:", font="Verdana 10")
        self.__configure_pred_lables()
        pass

    def __configure_window(self):
        self.master.title("Predict last name")
        self.master.geometry("450x300")
        self.master.resizable(0, 0)

    def __configure_pred_lables(self):
        self.result_0.place(x=230, y=100)
        self.result_1.place(x=230, y=120)
        self.result_2.place(x=230, y=140)

    def __predict(self):
        name = self.last_name.get()
        if len(name) == 0:
            self.result_0.configure(text="0:")
            self.result_1.configure(text="1:")
            self.result_2.configure(text="2:")
            return
        input = self.network.word_to_tensor(name)
        hidden = np.zeros((1, 128))
        for i in range(input.shape[0]):
            self.network.set_input(input[i])
            self.network.set_hidden(hidden)
            out = self.network.execute()
            out, hidden = out["output"], out["i2h"]
        sort_idx = np.argsort(out)
        self.result_0.configure(text="0:" + self.network.categories[sort_idx[0][0][0][-1]])
        self.result_1.configure(text="1:" + self.network.categories[sort_idx[0][0][0][-2]])
        self.result_2.configure(text="2:" + self.network.categories[sort_idx[0][0][0][-3]])

    def run(self):
        mainloop()
