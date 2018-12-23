import json
import os
import example.LeNet.model as model
import example.LeNet.gui as gui


class LeNetExample:
    def __init__(self):
        self.config = self.__load_config()
        self.model = model.Model(
            self.config["weights_folder"],
            self.config["dump_graph"])
        self.gui = gui.Gui(
            self.config["pensil_size"],
            self.config["main_window"],
            self.config["clear_window_button"],
            self.config["inference_button"],
        )

    def __load_config(self):
        config_file_name = "config.json"
        config_exists = os.path.isfile(config_file_name)
        if config_exists is False:
            raise Exception("[Error] Didnt found config file")
        with open(config_file_name) as f:
            return json.load(f)

    def run(self):
        self.gui.run(self.model)


if __name__ == "__main__":
    app = LeNetExample()
    app.run()