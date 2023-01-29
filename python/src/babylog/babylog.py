from __future__ import annotations

# needed because of issue https://github.com/python/cpython/issues/86813#issuecomment-1246097184
import concurrent.futures.thread
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
import os
import time
from typing import Optional, Dict, List


import boto3
import numpy as np


from babylog.config import Config
from babylog.data_utils import BoundingBoxDict
from babylog.logger import babylogger
from babylog.protobuf import (
    VisionModelType,
    VisionModel,
    DeviceDetails,
    Image,
    ImageBatch,
    ClassificationResult,
    BoundingBox,
    InferenceDevice,
    InferenceStats,
    ImagePrediction,
)
from babylog.pubsub import Publisher
from babylog.utils import (
    ndarray_to_Image,
    classification_from_dict,
    detection_from_dict,
)


class Babylog:
    def __init__(
        self,
        config_path: str,
        save_local: bool = True,
        save_cloud: bool = False,
        stream: bool = False,
    ):
        self.config = Config(config_path)
        self.save_local = save_local
        self.save_cloud = save_cloud
        self.stream = stream
        self._shutdown = False
        self._last_logged = time.perf_counter()
        if self.stream:
            self._publisher = Publisher(
                self.config.device.ip, self.config.device.ip, self.config.device.name
            )
        else:
            self._publisher = None
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.data_params.max_workers
        )
        babylogger.info(f"initialized babylog client")

    def log(self, *args, **kwargs):
        if (
            self.config is not None
            and (time.perf_counter() - self._last_logged) * 1000
            >= self.config.data_params.interval
        ):
            try:
                future = self.executor.submit(self._log, *args, **kwargs)
            except Exception as e:
                babylogger.error(f"could not submit logging job: {e}")
                raise ValueError(f"could not submit logging job: {e}")
            return future
        return None

    def _log(
        self,
        image: np.ndarray,
        model_type: VisionModelType,
        model_name: str,
        model_version: str,
        home_dir: str = "./babylog/",
        prediction: Optional[np.ndarray] = None,
        timestamp: Optional[int] = None,
        latency: Optional[int] = None,
        inference_device: Optional[InferenceDevice] = None,
        classification: Optional[Dict[str, float]] = None,
        detection: Optional[List[BoundingBoxDict]] = None,
        error_message: Optional[str] = "",
    ):
        try:
            single_image_prediction = ImagePrediction()
            single_image_prediction.raw_image.CopyFrom(ndarray_to_Image(array_=image))
            single_image_prediction.model.CopyFrom(
                VisionModel(type=model_type, version=model_version, name=model_name)
            )
            single_image_prediction.device_details.CopyFrom(
                DeviceDetails(
                    device_name=self.config.device.name,
                    group_name=self.config.device.group,
                )
            )

            if prediction is not None:
                single_image_prediction.prediction.CopyFrom(
                    ndarray_to_Image(array_=prediction)
                )

            if timestamp is not None:
                single_image_prediction.timestamp = timestamp
            else:
                single_image_prediction.timestamp = int(round(time.time() * 1000))

            if latency is not None or inference_device is not None:
                single_image_prediction.inference_stats.CopyFrom(
                    InferenceStats(
                        latency=latency,
                        inference_device=inference_device,
                        error_message=error_message,
                    )
                )

            if (
                model_type == VisionModelType.CLASSIFICATION
                and classification is not None
            ):
                single_image_prediction.classification_result.extend(
                    classification_from_dict(classification)
                )
            elif model_type == VisionModelType.DETECTION and detection is not None:
                single_image_prediction.bounding_boxes.extend(
                    detection_from_dict(detection)
                )
            else:
                raise NotImplementedError

            timestamp_save = datetime.now().strftime("%Y-%m-%d")
            timestamp_save_ms = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            group = self.config.device.group
            name = self.config.device.name
            dir_name = (
                f"{VisionModelType.Name(model_type)}/"
                f"{model_name}/{model_version}/{group}/{name}/"
                f"{timestamp_save}/"
            )
            filename = f"{timestamp_save_ms}.bin"
            prediction_binary = single_image_prediction.SerializeToString()

            if self.save_cloud:
                s3_storage = boto3.resource(
                    "s3",
                    aws_access_key_id=self.config.storage.aws_access_key_id,
                    aws_secret_access_key=self.config.storage.aws_secret_access_key,
                    region_name=self.config.storage.bucket_region,
                )
                s3_storage.Bucket(self.config.storage.bucket_name).put_object(
                    Key=f"{dir_name}{filename}", Body=prediction_binary
                )
                babylogger.info(f'successfully logged "{dir_name}{filename}" to cloud')

            if self.save_local:
                local_dir = f"{home_dir}{dir_name}"
                if not (os.path.isdir(local_dir)):
                    os.makedirs(local_dir, exist_ok=True)
                with open(f"{local_dir}{filename}", "wb") as f:
                    f.write(prediction_binary)
                babylogger.info(f'successfully logged "{local_dir}{filename}" locally')

            if self._publisher is not None:
                ret = self._publisher.send(prediction_binary)
                if ret:
                    babylogger.info(f"successfully streamed {filename}")
                else:
                    babylogger.error(f"could not stream {filename}")
            self._last_logged = time.perf_counter()
        except Exception as e:
            babylogger.error(f"could not log prediction: {e}")
            raise

    @property
    def shutdown(self):
        return self._shutdown

    def shutdown(self):
        babylogger.info(f"shutting down babylog client")
        if self._publisher is not None:
            self._publisher.shutdown()
        self._shutdown = True
        self.executor.shutdown(wait=True)
