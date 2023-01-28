from io import BytesIO
from typing import Dict, List

import numpy as np
import cv2

from babylog.data_utils import BoundingBoxDict
from babylog.protobuf import Image, ClassificationResult, BoundingBox


def bytes_to_image(raw_bytes_: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(raw_bytes_, np.byte), cv2.IMREAD_ANYCOLOR)


def image_to_bytes(array_: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", array_)[1].tobytes()


def ndarray_to_Image(array_: np.ndarray) -> Image:
    assert isinstance(array_, np.ndarray), "please pass a numpy array"
    assert len(array_.shape) == 3, "np.ndarray image has more than 3 dimensions"
    image = Image(
        **dict(
            zip(
                list(Image.DESCRIPTOR.fields_by_name.keys()),
                [*array_.shape, image_to_bytes(array_=array_)],
            )
        )
    )
    return image


def classification_from_dict(dict_: Dict[str, float]) -> List[ClassificationResult]:
    assert all(
        isinstance(key, str) for key in dict_.keys()
    ), "classification dict keys are not of type string"
    assert all(
        isinstance(val, float) for val in dict_.values()
    ), "classification dict values are not of type float"

    return [
        ClassificationResult(class_name=key_, probability=value_)
        for key_, value_ in dict_.items()
    ]


def check_bbox_dict(dict_: BoundingBoxDict):
    assert all(
        [elem in dict_ for elem in BoundingBoxDict.__annotations__.keys()]
    ), "incorrect format of the detection dictionary"
    assert all(
        [
            type(dict_[key_]) == value_
            for key_, value_ in BoundingBoxDict.__annotations__.items()
            if key_ != "classification"
        ]
    ), "incorrect format of the detection dictionary"


def detection_from_dict(bboxes: List[BoundingBoxDict]) -> List[BoundingBox]:
    detections = []
    for dict_ in bboxes:
        check_bbox_dict(dict_)
        assert set(dict_.keys()) == set(
            BoundingBoxDict.__annotations__.keys()
        ), "incorrect format of the detection dictionary"
        classifier_result_ = dict_["classification"]
        dict_.pop("classification")
        bboxes_message = BoundingBox(**dict_)
        bboxes_message.classification_result.extend(
            classification_from_dict(classifier_result_)
        )
        detections.append(bboxes_message)
    return detections
