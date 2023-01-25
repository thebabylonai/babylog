import babylog as bl
import cv2


bl.init('../resources/config.yaml')
img = cv2.imread('../resources/acr.png')
bl.log(image=img, prediction=img)
bl.shutdown()
