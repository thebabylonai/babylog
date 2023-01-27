import time

import cv2
from tqdm import tqdm

from babylog import Babylog, Image


# bl = Babylog('../resources/config.yaml')
img = cv2.imread('../resources/acr.png')
dd = list(img.shape)
dd.append('aa')
print(img.shape)
assert len(img.shape) == 3
# image = Image(**dict(zip(Image.__annotations__.keys(), dd)))
#
# print(dict(zip(list(Image.DESCRIPTOR.fields_by_name.keys()), [])))
# print(list(Image.DESCRIPTOR.fields_by_name.keys()))
# print(dd)
# for i in tqdm(range(10)):
#     bl.log(image=img, prediction=img)
#
# print('hey')
# # bl.shutdown()
