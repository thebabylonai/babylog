import babylog as bl
import cv2


bl.init('/home/airs-ahmad/PycharmProjects/babylog/resources/config.yaml')
img = cv2.imread('/home/airs-ahmad/PycharmProjects/babylog/resources/acr.png')
bl.log(image=img, prediction=img)
bl.shutdown()
