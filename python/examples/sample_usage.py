import time

import cv2
from tqdm import tqdm

from babylog import Babylog, VisionModelType, InferenceDevice


bl = Babylog('../resources/config.yaml', save_cloud=True)
img = cv2.imread('../resources/panda.jpg')


for i in tqdm(range(10)):
    bl.log(image=img,
           prediction=img,
           model_type=VisionModelType.DETECTION,
           model_name='best_model',
           model_version='1.0.0',
           compress=True,
           latency=500,
           inference_device=InferenceDevice.CPU,
           classification={'dog': 0.8, 'cat': 0.2},
           detection=([{'x': 0, 'y': 1, 'width': 100, 'height': 220, 'confidence': 0.97,
                       'classification': {'dog': 0.8, 'cat': 0.2}} for i in range(20)
                       ])
           )

