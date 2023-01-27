import time
from io import BytesIO
import yaml
from dataclasses import dataclass
from datetime import datetime
import threading
import queue
from typing import Optional, Dict, List, TypedDict
import pickle
# needed because of issue https://github.com/python/cpython/issues/86813#issuecomment-1246097184
import concurrent.futures.thread
from concurrent.futures.thread import ThreadPoolExecutor

import boto3
from botocore.config import Config as BotoConfig
import numpy as np
import cv2

from babylog.protobuf import VisionModelType, VisionModel, DeviceDetails, \
    Image, ImageBatch, ClassificationResult, SingleClassificationResult, BoundingBox,\
    InferenceDevice, InferenceStats, SingleImagePrediction


BoundingBoxes = List[BoundingBox]


@dataclass
class Device:
    ip: str
    name: str
    group: str


@dataclass
class DataParams:
    interval: int
    max_workers: int
    upload_batch_size: int


@dataclass
class StorageS3:
    aws_access_key_id: str
    aws_secret_access_key: str
    bucket_name: str
    bucket_region: str


@dataclass
class BabylonData:
    image: np.ndarray
    prediction: np.ndarray
    # desc: str
    # params: Dict


class BoundingBoxDict(TypedDict):
    x: int
    y: int
    width: int
    height: int
    confidence: float
    classification: Dict[str, float]


def ndarray_to_bytes(array_: np.ndarray) -> bytes:
    bytes_ = BytesIO()
    np.save(bytes_, array_, allow_pickle=False)
    return bytes_.getvalue()


def bytes_to_ndarray(raw_bytes_: bytes) -> np.ndarray:
    bytes_ = BytesIO(raw_bytes_)
    return np.load(bytes_, allow_pickle=False)


def ndarray_to_image(array_: np.ndarray) -> Image:
    assert isinstance(array_, np.ndarray), 'please pass a numpy array'
    assert len(array_.shape) == 3, 'np.ndarray image has more than 3 dimensions'
    image = Image(**dict(zip(list(Image.DESCRIPTOR.fields_by_name.keys()),
                             [*array_.shape, ndarray_to_bytes(array_)])))
    return image


def classification_from_dict(dict_: Dict[str, float]) -> ClassificationResult:
    assert all(isinstance(key, str) for key in dict_.keys()), 'classification dict keys are not of type string'
    assert all(isinstance(val, float) for val in dict_.values()), 'classification dict values are not of type float'
    classification = ClassificationResult()
    classification.classification_result.extend(
        [SingleClassificationResult(class_name=key_, probability=value_) for key_, value_ in dict_.items()])

    return classification


def check_bbox_dict(dict_: BoundingBoxDict):
    assert all([elem in dict_ for elem in BoundingBoxDict.__annotations__.keys()]), \
        'incorrect format of the detection dictionary'
    assert all([type(dict_[key_]) == value_
                for key_, value_ in BoundingBoxDict.__annotations__.items() if key_ != 'classification']), \
        'incorrect format of the detection dictionary'


def detection_from_dict(bbox_dict_: List[BoundingBoxDict]) -> BoundingBoxes:
    detections = BoundingBoxes()
    for dict_ in bbox_dict_:
        check_bbox_dict(dict_)
        [dict_.pop(kk) for kk in dict_.copy().keys() if kk not in BoundingBoxDict.__annotations__.keys()]
        dict_['classification_results'] = classification_from_dict(dict_.pop('classification'))

        detections.append(BoundingBox(**dict_))

    return detections


class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as filehandle:
            config = yaml.load(filehandle, Loader=yaml.FullLoader)
        print(config)
        self.device = Device(config['device']['ip'], config['device']['name'], config['device']['group'])
        self.data_params = DataParams(config['data']['interval'],
                                      config['data']['max_workers'],
                                      config['data']['upload_batch_size'])
        if config['S3_storage'] is None:
            raise NotImplementedError
        self.storage = StorageS3(config['S3_storage']['aws_access_key_id'],
                                config['S3_storage']['aws_secret_access_key'],
                                config['S3_storage']['bucket_name'],
                                config['S3_storage']['bucket_region'])


class Babylog:
    def __init__(self, config_path: str):
        self.config = Config(config_path)
        self.queue = queue.Queue(maxsize=self.config.data_params.upload_batch_size)
        self._shutdown = False

        self.executor = ThreadPoolExecutor(max_workers=self.config.data_params.max_workers)

    def log(self, *args, **kwargs):
        if self.config is not None and self.queue is not None:
            try:
                future = self.executor.submit(self._log, *args, **kwargs)
            except:
                raise ValueError('could not submit logging job ')
            return future
        return None

    def _log(self,
             image: np.ndarray,
             model_type: VisionModelType,
             model_name: str,
             model_version: str,
             prediction: Optional[np.ndarray] = None,
             timestamp: Optional[int] = None,
             latency: Optional[int] = None,
             inference_device: Optional[InferenceDevice] = None,
             classification: Optional[Dict[str, float]] = None,
             detection: Optional[List[BoundingBoxDict]] = None,
             error_message: Optional[str] = None
             ):
        print('Start')
        try:
            single_image_prediction = SingleImagePrediction()
            single_image_prediction.raw_image = ndarray_to_image(image)
            single_image_prediction.model = VisionModel(type=model_type, version=model_version, name=model_name)

            if prediction is not None:
                single_image_prediction.prediction = ndarray_to_bytes(prediction)

            if timestamp is not None:
                single_image_prediction.timestamp = timestamp
            else:
                single_image_prediction.timestamp = int(round(time.time() * 1000))

            if latency is not None or inference_device is not None:
                single_image_prediction.inference_stats = InferenceStats(latency=latency,
                                                                         inference_device=inference_device,
                                                                         error_message=error_message)

            if model_type == VisionModelType.CLASSIFICATION and classification is not None:
                single_image_prediction.classification = classification_from_dict(classification)
            elif model_type == VisionModelType.DETECTION and detection is not None:
                single_image_prediction.bounding_boxes.extend(detection_from_dict(detection))
            else:
                raise NotImplementedError

            s3_storage = boto3.resource(
                's3',
                aws_access_key_id=self.config.storage.aws_access_key_id,
                aws_secret_access_key=self.config.storage.aws_secret_access_key,
                region_name=self.config.storage.bucket_region
            )

            timestamp_save = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            s3_storage.Bucket(self.config.storage.bucket_name)\
                .put_object(Key=f'{model}/{timestamp_save}/raw.png', Body=single_image_prediction.SerializeToString())

            #
            #
            # img = cv2.imencode('.jpg', data.image)[1].tostring()
            # timestamp_save = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # name = self.config.device.name
            # s3_storage.Bucket(self.config.storage.bucket_name).put_object(Key=f'{name}/{timestamp_save}/raw.png', Body=img)
            #
            # pred_np = np.array(data.prediction)
            # pred_data = io.BytesIO()
            # pickle.dump(pred_np, pred_data)
            # pred_data.seek(0)
            # s3_storage.Bucket(self.config.storage.bucket_name).put_object(Key=f'{name}/{timestamp_save}/pred.pkl', Body=pred_data)
            # print('Done')
        except Exception as e:
            print(e)


    @property
    def shutdown(self):
        return self._shutdown

    def shutdown(self):
        while True:
            if self.queue.qsize() <= 0:
                self._shutdown = True
                break
        self.executor.shutdown(wait=True)





