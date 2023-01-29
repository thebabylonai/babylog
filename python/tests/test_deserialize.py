import pytest
import cv2
import os
import shutil
from datetime import datetime
from freezegun import freeze_time

from babylog import Babylog, VisionModelType, InferenceDevice, LoggedPrediction
from babylog.utils import image_to_bytes


def teardown_module(module):
    """Clean up the logs"""
    shutil.rmtree("./babylog/")


class TestDeserialize:
    def setup_class(self):
        self.bl = Babylog("./python/resources/config.yaml")
        self.img = cv2.imread("./python/resources/panda.jpg")

    @freeze_time("2023-01-29 03:21:34", tz_offset=0)
    def test_serialized_image(self):
        """Test to see that an object gets serialized at the right time"""
        self.bl.log(
            image=self.img,
            prediction=self.img,
            model_type=VisionModelType.DETECTION,
            model_name="test_model",
            model_version="1.0.0",
            latency=500,
            inference_device=InferenceDevice.CPU,
            classification={"dog": 0.8, "cat": 0.2},
            detection=(
                [
                    {
                        "x": 0,
                        "y": 1,
                        "width": 100,
                        "height": 220,
                        "confidence": 0.97,
                        "classification": {"dog": 0.8, "cat": 0.2},
                    }
                    for i in range(3)
                ]
            ),
        )

        self.bl.shutdown()
        assert os.path.exists(
            f"./babylog/DETECTION/test_model/1.0.0/GROUP_NAME/DEVICE_NAME/{datetime.now().strftime('%Y-%m-%d')}/2023-01-29 03:21:34.000.bin"
        )

    @pytest.fixture
    def get_prediction(self):
        prediction = LoggedPrediction.from_path(
            "./babylog/DETECTION/test_model/1.0.0/GROUP_NAME/DEVICE_NAME/2023-01-29/2023-01-29 03:21:34.000.bin"
        )
        return prediction

    def test_deserialized_prediction_model(self, get_prediction):
        deserialized_model = get_prediction.model
        pre_serialized_model = {
            "type": "DETECTION",
            "name": "test_model",
            "version": "1.0.0",
        }
        assert deserialized_model == pre_serialized_model

    def test_deserialized_prediction_inference_stats(self, get_prediction):
        deserialized_model = get_prediction.inference_stats
        pre_serialized_model = {
            "latency": 500,
            "inferenceDevice": "CPU",
            "errorMessage": "",
        }
        assert deserialized_model == pre_serialized_model

    def test_deserialized_prediction_image(self, get_prediction):
        deserialized_image = get_prediction._prediction.raw_image.image_bytes
        assert deserialized_image == image_to_bytes(self.img)
