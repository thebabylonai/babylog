from dataclasses import dataclass

import yaml


@dataclass
class Device:
    ip: str
    name: str
    group: str


@dataclass
class DataParams:
    interval: int
    max_workers: int


@dataclass
class StorageS3:
    aws_access_key_id: str
    aws_secret_access_key: str
    bucket_name: str
    bucket_region: str


class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as filehandle:
            config = yaml.load(filehandle, Loader=yaml.FullLoader)
        print(config)
        self.device = Device(config['device']['ip'], config['device']['name'], config['device']['group'])
        self.data_params = DataParams(config['data']['interval'],
                                      config['data']['max_workers'])
        if config['S3_storage'] is None:
            raise NotImplementedError
        self.storage = StorageS3(config['S3_storage']['aws_access_key_id'],
                                config['S3_storage']['aws_secret_access_key'],
                                config['S3_storage']['bucket_name'],
                                config['S3_storage']['bucket_region'])

