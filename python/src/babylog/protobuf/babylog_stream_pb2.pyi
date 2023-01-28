from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

CLASSIFICATION: VisionModelType
CPU: InferenceDevice
CUDA: InferenceDevice
DEFAULT: InferenceDevice
DESCRIPTOR: _descriptor.FileDescriptor
DETECTION: VisionModelType
GPU: InferenceDevice
NONE: VisionModelType

class BoundingBox(_message.Message):
    __slots__ = ["classification_result", "confidence", "height", "width", "x", "y"]
    CLASSIFICATION_RESULT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    classification_result: _containers.RepeatedCompositeFieldContainer[
        ClassificationResult
    ]
    confidence: float
    height: int
    width: int
    x: int
    y: int
    def __init__(
        self,
        x: _Optional[int] = ...,
        y: _Optional[int] = ...,
        width: _Optional[int] = ...,
        height: _Optional[int] = ...,
        confidence: _Optional[float] = ...,
        classification_result: _Optional[
            _Iterable[_Union[ClassificationResult, _Mapping]]
        ] = ...,
    ) -> None: ...

class ClassificationResult(_message.Message):
    __slots__ = ["class_name", "probability"]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    class_name: str
    probability: float
    def __init__(
        self, class_name: _Optional[str] = ..., probability: _Optional[float] = ...
    ) -> None: ...

class DeviceDetails(_message.Message):
    __slots__ = ["device_name", "group_name"]
    DEVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    GROUP_NAME_FIELD_NUMBER: _ClassVar[int]
    device_name: str
    group_name: str
    def __init__(
        self, device_name: _Optional[str] = ..., group_name: _Optional[str] = ...
    ) -> None: ...

class Image(_message.Message):
    __slots__ = ["channels", "height", "image_bytes", "width"]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    channels: int
    height: int
    image_bytes: bytes
    width: int
    def __init__(
        self,
        height: _Optional[int] = ...,
        width: _Optional[int] = ...,
        channels: _Optional[int] = ...,
        image_bytes: _Optional[bytes] = ...,
    ) -> None: ...

class ImageBatch(_message.Message):
    __slots__ = ["batch_size", "channels", "height", "image_bytes", "width"]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    batch_size: int
    channels: int
    height: int
    image_bytes: bytes
    width: int
    def __init__(
        self,
        height: _Optional[int] = ...,
        width: _Optional[int] = ...,
        channels: _Optional[int] = ...,
        batch_size: _Optional[int] = ...,
        image_bytes: _Optional[bytes] = ...,
    ) -> None: ...

class ImagePrediction(_message.Message):
    __slots__ = [
        "bounding_boxes",
        "classification_result",
        "device_details",
        "inference_stats",
        "model",
        "prediction",
        "raw_image",
        "timestamp",
    ]
    BOUNDING_BOXES_FIELD_NUMBER: _ClassVar[int]
    CLASSIFICATION_RESULT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_DETAILS_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STATS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_FIELD_NUMBER: _ClassVar[int]
    RAW_IMAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    bounding_boxes: _containers.RepeatedCompositeFieldContainer[BoundingBox]
    classification_result: _containers.RepeatedCompositeFieldContainer[
        ClassificationResult
    ]
    device_details: DeviceDetails
    inference_stats: InferenceStats
    model: VisionModel
    prediction: Image
    raw_image: Image
    timestamp: int
    def __init__(
        self,
        timestamp: _Optional[int] = ...,
        device_details: _Optional[_Union[DeviceDetails, _Mapping]] = ...,
        model: _Optional[_Union[VisionModel, _Mapping]] = ...,
        raw_image: _Optional[_Union[Image, _Mapping]] = ...,
        prediction: _Optional[_Union[Image, _Mapping]] = ...,
        inference_stats: _Optional[_Union[InferenceStats, _Mapping]] = ...,
        classification_result: _Optional[
            _Iterable[_Union[ClassificationResult, _Mapping]]
        ] = ...,
        bounding_boxes: _Optional[_Iterable[_Union[BoundingBox, _Mapping]]] = ...,
    ) -> None: ...

class InferenceStats(_message.Message):
    __slots__ = ["error_message", "inference_device", "latency"]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_DEVICE_FIELD_NUMBER: _ClassVar[int]
    LATENCY_FIELD_NUMBER: _ClassVar[int]
    error_message: str
    inference_device: InferenceDevice
    latency: int
    def __init__(
        self,
        latency: _Optional[int] = ...,
        inference_device: _Optional[_Union[InferenceDevice, str]] = ...,
        error_message: _Optional[str] = ...,
    ) -> None: ...

class VisionModel(_message.Message):
    __slots__ = ["name", "type", "version"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: VisionModelType
    version: str
    def __init__(
        self,
        type: _Optional[_Union[VisionModelType, str]] = ...,
        version: _Optional[str] = ...,
        name: _Optional[str] = ...,
    ) -> None: ...

class VisionModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class InferenceDevice(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
