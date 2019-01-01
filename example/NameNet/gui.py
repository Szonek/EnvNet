from tkinter import *
import numpy as np

class Gui:
    """
    This class defines the simple GUI to present NameNet model.
    """
    def __init__(self, network):
        self.network = network
        self.master = Tk()
        self.last_name = Entry(self.master)
        self.last_name.grid(row=0, column=1)
        Label(self.master, text="Last name:").grid(row=0)
        Button(self.master, text="Predict", command=self.__predict).grid(row=3, column=0, sticky=W, pady=4)
        pass

    def __predict(self):
        input = self.network.word_to_tensor(self.last_name.get())
        hidden = np.zeros((1, 128))
        for i in range(input.shape[0]):
            self.network.set_input(input[i])
            self.network.set_hidden(hidden)
            out = self.network.execute()
            out, hidden = out["output"], out["i2h"]
        sort_idx = np.argsort(out)
        category_i = sort_idx[0][0][0][-1]
        print(self.network.categories[category_i], category_i)
        category_i = sort_idx[0][0][0][-2]
        print(self.network.categories[category_i], category_i)
        category_i = sort_idx[0][0][0][-3]
        print(self.network.categories[category_i], category_i)

    def run(self):
        mainloop()
