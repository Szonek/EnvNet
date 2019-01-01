import json
import os
import example.NameNet.model as model
import example.NameNet.gui as gui


class NameNetExample:
    """
    Creates and handles the example.
    Proper file "config.json" should be located in the same dict as this file.
    """
    def __init__(self):
        self.config = self.__load_config()
        self.model = model.Model(
            self.config["weights_folder"],
            self.config["dump_graph"])
        self.gui = gui.Gui(self.model)

    def __load_config(self):
        config_file_name = "config.json"
        config_exists = os.path.isfile(config_file_name)
        if config_exists is False:
            raise Exception("[Error] Didnt found config file")
        with open(config_file_name) as f:
            return json.load(f)

    def run(self):
        self.gui.run()


if __name__ == "__main__":
    app = NameNetExample()
    app.run()