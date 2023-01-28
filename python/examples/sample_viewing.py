import cv2

from babylog import LoggedPrediction


my_prediction = LoggedPrediction.from_path(
    "/home/aroumie/Documents/BabylonAI/babylog/python/examples/babylog/DETECTION/best_model/1.0.0/GROUP_NAME/3d48b246-94f0-4e62-8231-e3f5b471db0c/2023-01-28/2023-01-28 18:52:21.438.bin"
)
# print(my_prediction.model)
# print(my_prediction.inference_stats)
# print(my_prediction.image)
# print(my_prediction.prediction)

# cv2.imshow('image', my_prediction.prediction)
# cv2.waitKey(0)
# print(my_prediction._prediction.bounding_boxes)
print(my_prediction.detection)
# print(my_prediction.detection)
