import time

import cv2
from tqdm import tqdm
import numpy as np

from babylog import Babylog, VisionModelType, InferenceDevice


bl = Babylog('../resources/config.yaml', save_cloud=True, stream=True)
img = cv2.imread('../resources/acr.png')


for i in tqdm(range(100)):
    bl.log(image=img,
           prediction=img,
           model_type=VisionModelType.DETECTION,
           model_name='best_model',
           model_version='1.0.0',
           latency=500,
           inference_device=InferenceDevice.CPU,
           classification={'dog': 0.8, 'cat': 0.2},
           detection=([{'x': 0, 'y': 1, 'width': 100, 'height': 220, 'confidence': 0.97,
                       'classification': {'dog': 0.8, 'cat': 0.2}} for i in range(3)
                       ])
           )
    time.sleep(1)

