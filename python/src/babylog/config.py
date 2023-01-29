from dataclasses import dataclass

import yaml

from babylog.logger import babylogger


@dataclass
class Device:
    ip: str
    port: int
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
        with open(config_path, "r") as filehandle:
            config = yaml.load(filehandle, Loader=yaml.FullLoader)
        babylogger.info(f"babylog.config contents: ")
        babylogger.info(f"{config}")
        try:
            self.device = Device(
                config["device"]["ip"],
                config["device"]["port"],
                config["device"]["name"],
                config["device"]["group"],
            )
            self.data_params = DataParams(
                config["data"]["interval"], config["data"]["max_workers"]
            )
            if config["S3_storage"] is None:
                raise NotImplementedError
            self.storage = StorageS3(
                config["S3_storage"]["aws_access_key_id"],
                config["S3_storage"]["aws_secret_access_key"],
                config["S3_storage"]["bucket_name"],
                config["S3_storage"]["bucket_region"],
            )
        except Exception as e:
            babylogger.error(f"could not properly setup config: {e}")
            raise
