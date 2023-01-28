from typing import Dict, List

from google.protobuf.json_format import MessageToJson, MessageToDict
import numpy as np


from babylog.protobuf import ImagePrediction, VisionModel, InferenceStats
from babylog.utils import bytes_to_image


class LoggedPrediction:
    def __init__(self, binary_: bytes):
        self._prediction = ImagePrediction()
        self._prediction.ParseFromString(binary_)

    @property
    def model(self) -> Dict:
        return MessageToDict(self._prediction.model)

    @property
    def inference_stats(self) -> Dict:
        return MessageToDict(self._prediction.inference_stats)

    @property
    def device_details(self) -> Dict:
        return MessageToDict(self._prediction.device_details)

    @property
    def classification(self) -> List[Dict]:
        return [MessageToDict(res) for res in self._prediction.classification_result]

    @property
    def detection(self) -> List[Dict]:
        return [MessageToDict(bbox) for bbox in self._prediction.bounding_boxes]

    @property
    def image(self) -> np.ndarray:
        return bytes_to_image(self._prediction.raw_image.image_bytes)

    @property
    def prediction(self) -> np.ndarray:
        return bytes_to_image(self._prediction.prediction.image_bytes)

    @property
    def timestamp(self) -> int:
        return self.prediction.timestamp

    @classmethod
    def from_path(cls, filepath: str):
        with open(filepath, "rb") as f:
            binary_ = f.read()
        return cls(binary_)

    @classmethod
    def from_bytes(cls, binary_: bytes):
        return cls(binary_)
