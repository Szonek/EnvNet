import cv2
import numpy as np
import scipy.ndimage.interpolation as sp


class Gui:
    def __init__(self, pensil_size, window_opts,
                 clear_button, inference_button):
        self.pensil_size = pensil_size
        self.main_window_name = "main_window"
        self.small_window_name = "small_window"
        self.main_window = self.__define_main_window(window_opts)
        self.drawing, self.mode, self.ix, self.iy = False, False, -1, -1
        self.clear = clear_button
        self.inference = inference_button
        cv2.setMouseCallback(self.main_window_name, self.__draw_circle)

    def __draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.main_window, (x, y), self.pensil_size, (255, 255, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.circle(self.main_window, (x, y), self.pensil_size, (255, 255, 255), -1)

    def __define_main_window(self, window_opts):
        cv2.namedWindow(self.main_window_name)
        cv2.resizeWindow(self.main_window_name, window_opts["width"], window_opts["height"])
        cv2.moveWindow(self.main_window_name, window_opts["start_x"], window_opts["start_y"])
        return np.zeros((window_opts["width"], window_opts["height"], 1), np.float)

    def run(self, network):
        real_inp_size = network.get_input_size()
        ratio = real_inp_size / self.main_window.shape[0]
        while(1):
            cv2.imshow(self.main_window_name, self.main_window)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(self.clear):
                self.main_window = np.zeros(self.main_window.shape, np.float)
            if k == ord(self.inference):
                img_to_network = sp.zoom(self.main_window, (ratio, ratio, 1))
                img_to_network = np.reshape(img_to_network, (1, 1, real_inp_size, real_inp_size), order='C')
                img_to_network /= 255
                network.set_input(img_to_network)
                out = network.execute()
                print(np.argmax(out['output']))
            if k == 27:
                break
        cv2.destroyAllWindows()