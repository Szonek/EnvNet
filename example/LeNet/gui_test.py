import cv2
import numpy as np
import scipy.ndimage.interpolation as sp
import example.LeNet.lenet

network = example.LeNet.lenet.LeNet()

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),2,(255,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),2,(255,255,255),-1)

img = np.zeros((128,128,1), np.float)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)


while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        img = np.zeros((128, 128, 1), np.float)
    if k == ord('a'):
        img_to_network = sp.zoom(img, (1/4, 1/4, 1))
        img_to_network /= 255
        network.set_input(img_to_network)
        out = network.execute()
        print('end')
    elif k == 27:
        break


print("end test")
cv2.destroyAllWindows()