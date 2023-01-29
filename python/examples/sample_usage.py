import time

import cv2
from tqdm import tqdm
import numpy as np

from babylog import Babylog, VisionModelType, InferenceDevice


bl = Babylog("../resources/config.yaml", save_cloud=False, stream=False)
img = cv2.imread("../resources/panda.jpg")


for i in tqdm(range(100)):
    bl.log(
        image=img,
        prediction=img,
        model_type=VisionModelType.DETECTION,
        model_name="resnet50_finetuned",
        model_version="1.0.0",
        latency=10,
        inference_device=InferenceDevice.CPU,
        detection=(
            [
                {
                    "x": 1,
                    "y": 1,
                    "width": 100,
                    "height": 220,
                    "confidence": 0.97,
                    "classification": {"dog": 0.8, "cat": 0.2},
                },
                {
                    "x": 1,
                    "y": 1,
                    "width": 50,
                    "height": 150,
                    "confidence": 0.97,
                    "classification": {"dog": 0.5, "cat": 0.5},
                },
            ]
        ),
    )
    time.sleep(1)
