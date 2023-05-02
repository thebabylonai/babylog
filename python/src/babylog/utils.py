from io import BytesIO
import os
from typing import Dict, List

import numpy as np
import cv2
from tqdm import tqdm

from babylog.data_utils import BoundingBoxDict
from babylog.protobuf import Image, ClassificationResult, BoundingBox
from babylog.logger import babylogger


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


class ImageSequence:
    def __init__(self, path=None, extensions=None, timestamp_file=None, 
                 rotate=False, scale=0.5, **kwargs):
        self.path = path
        self.rotate = rotate
        self.scale = scale
        self.width = None
        self.height = None
        self.frame_count = None
        self.video = None
        self.images = None
        self.scaled_dim = None
        self.frame_num = 0
        self.playback_speed = 50
        self.raw_video_string = "Scaled Raw Image"
        self.frame = None
        self.is_image = False
        self.is_video = False

        if timestamp_file is None:
            self.timestamps = None
        else:
            with open(timestamp_file, 'r') as filehandle:
                self.timestamps = [timestamp.rstrip() for timestamp in filehandle.readlines()]

        if self.path is not None:
            if self.check_directory():
                self.is_video = False
                if not extensions:
                    extensions = [".jpg", ".jpeg", ".png"]
                self.images = [f for f in sorted(os.listdir(self.path)) if os.path.splitext(f)[1] in extensions]
                babylogger.info('Found {} images in folder'.format(len(self.images)))
                if not self.images:
                    self.is_image = False
                    raise NotImplementedError
                else:
                    self.is_image = True
            elif self.check_video():
                self.is_video = True
            else:
                raise NotImplementedError

            if self.is_video:
                self.video = cv2.VideoCapture(self.path)

        self.get_dimensions()
        self.scaled_dim = (int(self.width * self.scale),
                           int(self.height * self.scale))
        config = {'video': self.is_video, 'image': self.is_image,
                  'scaled dimensions': self.scaled_dim,
                  'frame count': self.frame_count}
        babylogger.info('[CONFIG video]: {}'.format(config))
        super().__init__(**kwargs)

    def check_directory(self):
        return os.path.isdir(self.path)

    def check_file(self):
        return os.path.isfile(self.path)

    def check_video(self):
        extensions = [".mp4", ".avi"]
        if os.path.splitext(self.path)[1] in extensions:
            return True
        return False

    def get_dimensions(self):
        if self.is_video:
            width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.width = width if not self.rotate else height
            self.height = height if not self.rotate else width
            self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.frame_count <= 0:
                self.frame_count = 0
                while True:
                    ret, _ = self.video.read()
                    if not ret:
                        break
                    self.frame_count += 1
                self.video.release()
                self.video = cv2.VideoCapture(self.path)

        if self.is_image:
            image = cv2.imread(self.path + '/' + self.images[0])
            self.width = image.shape[1]
            self.height = image.shape[0]
            self.frame_count = len(self.images)

    def get_frame(self, frame_num):
        self.frame_num = frame_num
        if self.is_video:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.video.read()
            if not ret:
                frame = None
                babylogger.info('Could not read frame')
            else:
                pass

        if self.is_image:
            image_path = self.images[frame_num]
            try:
                frame = cv2.imread(self.path + '/' + image_path)
            except Exception as e:
                babylogger.error('{e}'.format(e))
                frame = None

        try:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) if self.rotate else frame
            frame = cv2.resize(frame, self.scaled_dim, interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            babylogger.error('Invalid frame!')
            frame = None
            pass

        self.frame = frame

    def check_frames(self, max_frames=300):
      if self.frame_count > max_frames:
        return False
      return True

    def get_frames(self):
      for i in tqdm(range(self.frame_count)):
        self.get_frame(i)
        yield self.frame