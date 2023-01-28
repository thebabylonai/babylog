# needed because of issue https://github.com/python/cpython/issues/86813#issuecomment-1246097184
import concurrent.futures.thread
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime

import time
from typing import Optional, Dict, List


import boto3
import numpy as np


from babylog.config import Config
from babylog.data_utils import BoundingBoxDict
from babylog.logger import babylogger
from babylog.protobuf import VisionModelType, VisionModel, DeviceDetails, \
    Image, ImageBatch, ClassificationResult, BoundingBox,\
    InferenceDevice, InferenceStats, SingleImagePrediction
from babylog.utils import ndarray_to_image, ndarray_to_bytes, classification_from_dict, \
    detection_from_dict


class Babylog:
    def __init__(self, config_path: str):
        self.config = Config(config_path)
        self._shutdown = False
        self.executor = ThreadPoolExecutor(max_workers=self.config.data_params.max_workers)
        babylogger.info(f'initialized babylog client')

    def log(self, *args, **kwargs):
        if self.config is not None:
            try:
                future = self.executor.submit(self._log, *args, **kwargs)
            except Exception as e:
                babylogger.error(f'could not submit logging job: {e}')
                raise ValueError(f'could not submit logging job: {e}')
            return future
        return None

    def _log(self,
             image: np.ndarray,
             model_type: VisionModelType,
             model_name: str,
             model_version: str,
             compress: Optional[bool] = True,
             prediction: Optional[np.ndarray] = None,
             timestamp: Optional[int] = None,
             latency: Optional[int] = None,
             inference_device: Optional[InferenceDevice] = None,
             classification: Optional[Dict[str, float]] = None,
             detection: Optional[List[BoundingBoxDict]] = None,
             error_message: Optional[str] = None
             ):
        try:
            single_image_prediction = SingleImagePrediction()
            single_image_prediction.raw_image.CopyFrom(ndarray_to_image(array_=image, compress=compress))
            single_image_prediction.model.CopyFrom(VisionModel(type=model_type, version=model_version, name=model_name))

            if prediction is not None:
                single_image_prediction.prediction = ndarray_to_bytes(array_=prediction, compress=compress)

            if timestamp is not None:
                single_image_prediction.timestamp = timestamp
            else:
                single_image_prediction.timestamp = int(round(time.time() * 1000))

            if latency is not None or inference_device is not None:
                single_image_prediction.inference_stats.CopyFrom(InferenceStats(latency=latency,
                                                                                inference_device=inference_device,
                                                                                error_message=error_message))

            if model_type == VisionModelType.CLASSIFICATION and classification is not None:
                single_image_prediction.classification_result.extend(classification_from_dict(classification))
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
            timestamp_save = datetime.now().strftime('%Y-%m-%d')
            timestamp_save_ms = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            group = self.config.device.group
            name = self.config.device.name
            filename = f'{VisionModelType.Name(model_type)}/' \
                       f'{model_name}/{model_version}/{group}/{name}/'\
                       f'{timestamp_save}/{timestamp_save_ms}'
            s3_storage.Bucket(self.config.storage.bucket_name)\
                .put_object(Key=filename,
                            Body=single_image_prediction.SerializeToString())
            babylogger.info(f'successfully logged {filename}')
        except Exception as e:
            babylogger.error(f'could not log prediction: {e}')
            raise ValueError(f'could not log prediction: {e}')

    @property
    def shutdown(self):
        return self._shutdown

    def shutdown(self):
        babylogger.info(f'shutting down babylog client')
        self._shutdown = True
        self.executor.shutdown(wait=True)





