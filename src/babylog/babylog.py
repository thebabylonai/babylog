import time
import io
import yaml
from dataclasses import dataclass
from datetime import datetime
import threading
import queue
from typing import Optional, Dict
import pickle

import boto3
from botocore.config import Config as BotoConfig
import numpy as np
import cv2

@dataclass
class Device:
    ip: str
    uuid: str
    group: str


@dataclass
class DataParams:
    interval: int
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

#TODO: S3 writer class
# class S3Writer:

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as filehandle:
            config = yaml.load(filehandle, Loader=yaml.FullLoader)
        print(config)
        self.device = Device(config['device']['ip'], config['device']['uuid'], config['device']['group'])
        self.data_params = DataParams(config['data']['interval'], config['data']['upload_batch_size'])
        if config['S3_storage'] is None:
            raise NotImplementedError
        self.storage = StorageS3(config['S3_storage']['aws_access_key_id'],
                                config['S3_storage']['aws_secret_access_key'],
                                config['S3_storage']['bucket_name'],
                                config['S3_storage']['bucket_region'])


class Babylog:
    def __init__(self):
        self.config = None
        self.queue = None
        self.lock = threading.Lock()
        self._shutdown = False
        threading.Thread(target=self.logging).start()

    def init(self, config_path: str):
        self.config = Config(config_path)
        self.queue = queue.Queue(maxsize=self.config.data_params.upload_batch_size)

    def log(self, image: np.ndarray, prediction: np.ndarray): #, desc: Optional[str], params: Optional[Dict]
        if self.config is not None and self.queue is not None:
            # if desc is None:
            #     desc = ''
            # if params is None:
            #     params = {}
            self.queue.put(BabylonData(image, prediction)) #, desc, params

    def logging(self):
        while not self._shutdown:
            if self.queue is not None:
                # if len(self.queue) == self.config.data_params.upload_batch_size:
                data = self.queue.get()
                s3_storage = boto3.resource(
                    's3',
                    aws_access_key_id=self.config.storage.aws_access_key_id,
                    aws_secret_access_key=self.config.storage.aws_secret_access_key,
                    region_name=self.config.storage.bucket_region
                )
                img = cv2.imencode('.jpg', data.image)[1].tostring()
                timestamp_save = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                uuid = self.config.device.uuid
                s3_storage.Bucket(self.config.storage.bucket_name).put_object(Key=f'{uuid}/{timestamp_save}/raw.png', Body=img)

                pred_np = np.array(data.prediction)
                pred_data = io.BytesIO()
                pickle.dump(pred_np, pred_data)
                pred_data.seek(0)
                s3_storage.Bucket(self.config.storage.bucket_name).put_object(Key=f'{uuid}/{timestamp_save}/pred.pkl', Body=pred_data)



    @property
    def shutdown(self):
        return self._shutdown

    def shutdown(self):
        while True:
            if self.queue.qsize() <= 0:
                self._shutdown = True
                break


babylog = Babylog()


def init(config: str):
    babylog.init(config)

def log(image: np.ndarray, prediction: np.ndarray):
    babylog.log(image, prediction)

def shutdown():
    babylog.shutdown()
# def log(image: np.ndarray, prediction: np.ndarray, desc: Optional[str], params: Optional[Dict]):
#     babylog.log(image, prediction, desc if desc is not None else '', params if params is not None else {})





