#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import Optional, Dict, Any

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object


if HAS_PYDANTIC:
    class RecognitionRecord(BaseModel):
        id: Optional[str] = None
        user_id: str
        username: str
        image_filename: str
        image_path: str
        recognition_result: Dict[str, Any]
        timestamp: datetime = datetime.now()
        model_used: str
        processing_time: float
        is_multi_role: bool
        nsfw_status: bool
        detected_text: bool

    class RecognitionRecordCreate(BaseModel):
        user_id: str
        username: str
        image_filename: str
        image_path: str
        recognition_result: Dict[str, Any]
        model_used: str
        processing_time: float
        is_multi_role: bool
        nsfw_status: bool
        detected_text: bool

    class RecognitionRecordResponse(BaseModel):
        id: str
        user_id: str
        username: str
        image_filename: str
        image_path: str
        recognition_result: Dict[str, Any]
        timestamp: datetime
        model_used: str
        processing_time: float
        is_multi_role: bool
        nsfw_status: bool
        detected_text: bool

        class Config:
            from_attributes = True
else:
    class RecognitionRecord:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id")
            self.user_id = kwargs.get("user_id", "")
            self.username = kwargs.get("username", "")
            self.image_filename = kwargs.get("image_filename", "")
            self.image_path = kwargs.get("image_path", "")
            self.recognition_result = kwargs.get("recognition_result", {})
            self.timestamp = kwargs.get("timestamp", datetime.now())
            self.model_used = kwargs.get("model_used", "")
            self.processing_time = kwargs.get("processing_time", 0.0)
            self.is_multi_role = kwargs.get("is_multi_role", False)
            self.nsfw_status = kwargs.get("nsfw_status", False)
            self.detected_text = kwargs.get("detected_text", False)

    class RecognitionRecordCreate:
        def __init__(self, **kwargs):
            self.user_id = kwargs.get("user_id", "")
            self.username = kwargs.get("username", "")
            self.image_filename = kwargs.get("image_filename", "")
            self.image_path = kwargs.get("image_path", "")
            self.recognition_result = kwargs.get("recognition_result", {})
            self.model_used = kwargs.get("model_used", "")
            self.processing_time = kwargs.get("processing_time", 0.0)
            self.is_multi_role = kwargs.get("is_multi_role", False)
            self.nsfw_status = kwargs.get("nsfw_status", False)
            self.detected_text = kwargs.get("detected_text", False)

    class RecognitionRecordResponse:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", "")
            self.user_id = kwargs.get("user_id", "")
            self.username = kwargs.get("username", "")
            self.image_filename = kwargs.get("image_filename", "")
            self.image_path = kwargs.get("image_path", "")
            self.recognition_result = kwargs.get("recognition_result", {})
            self.timestamp = kwargs.get("timestamp", datetime.now())
            self.model_used = kwargs.get("model_used", "")
            self.processing_time = kwargs.get("processing_time", 0.0)
            self.is_multi_role = kwargs.get("is_multi_role", False)
            self.nsfw_status = kwargs.get("nsfw_status", False)
            self.detected_text = kwargs.get("detected_text", False)
