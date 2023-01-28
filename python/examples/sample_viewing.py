import cv2

from babylog import LoggedPrediction


my_prediction = LoggedPrediction.from_path(
    "/home/aroumie/Documents/BabylonAI/babylog/python/examples/babylog/DETECTION/best_model/1.0.0/GROUP_NAME/3d48b246-94f0-4e62-8231-e3f5b471db0c/2023-01-28/2023-01-28 18:52:21.438.bin"
)

print(my_prediction.detection)
