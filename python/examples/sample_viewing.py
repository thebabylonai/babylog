import cv2

from babylog import LoggedPrediction


my_prediction = LoggedPrediction.from_path(
    "PATH_TO_PREDICTION"
)

print(my_prediction.detection)
