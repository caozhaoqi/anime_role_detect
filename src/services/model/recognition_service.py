#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别记录服务

管理用户的识别历史记录
支持数据库和JSON文件两种存储方式
"""

import json
import os
from datetime import datetime
from typing import List, Optional
from src.models.recognition_record import RecognitionRecord, RecognitionRecordCreate
from src.core.logging.global_logger import get_logger

logger = get_logger("recognition_service")

USE_DATABASE = os.environ.get("USE_DATABASE", "true").lower() == "true"
_use_database = None


def _init_db():
    """初始化数据库"""
    global _use_database
    if _use_database is None:
        _use_database = USE_DATABASE
        if _use_database:
            try:
                from src.core.config.database import init_database, create_tables

                init_database()
                create_tables()
                logger.info("数据库存储模式已启用")
            except Exception as e:
                logger.error(f"数据库初始化失败: {e}")
                _use_database = False


class RecognitionService:
    """识别记录服务"""

    def __init__(self):
        """初始化识别记录服务"""
        _init_db()
        self.records_file = os.path.abspath("data/recognition_records.json")
        # 确保数据目录存在
        os.makedirs(os.path.dirname(self.records_file), exist_ok=True)
        self.records = self._load_records() if not _use_database else []

    def _load_records(self) -> List[RecognitionRecord]:
        """从文件加载识别记录"""
        try:
            if os.path.exists(self.records_file):
                with open(self.records_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    records = []
                    for item in data:
                        if "timestamp" in item:
                            item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                        records.append(RecognitionRecord(**item))
                    logger.info(f"从文件加载了 {len(records)} 条识别记录")
                    return records
            return []
        except Exception as e:
            logger.error(f"加载识别记录失败: {e}")
            return []

    def _save_records(self):
        """保存识别记录到文件"""
        try:
            data = []
            for record in self.records:
                record_dict = record.model_dump()
                if "timestamp" in record_dict and isinstance(record_dict["timestamp"], datetime):
                    record_dict["timestamp"] = record_dict["timestamp"].isoformat()
                data.append(record_dict)

            with open(self.records_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"保存了 {len(self.records)} 条识别记录")
        except Exception as e:
            logger.error(f"保存识别记录失败: {e}")

    def create_record(self, record: RecognitionRecordCreate) -> RecognitionRecord:
        """创建新的识别记录"""
        record_id = f"rec_{int(datetime.now().timestamp() * 1000)}"

        if _use_database:
            try:
                from src.services.support.database_service import RecognitionRecordDB, get_db_service

                db = get_db_service()
                db_record = RecognitionRecordDB.create(
                    db=db,
                    record_id=record_id,
                    user_id=record.user_id,
                    username=record.username,
                    image_filename=record.image_filename,
                    image_path=record.image_path,
                    recognition_result=record.recognition_result,
                    model_used=record.model_used,
                    processing_time=record.processing_time,
                    is_multi_role=record.is_multi_role,
                    nsfw_status=record.nsfw_status,
                    detected_text=record.detected_text,
                )
                new_record = RecognitionRecord(
                    id=db_record.id,
                    user_id=db_record.user_id,
                    username=db_record.username,
                    image_filename=db_record.image_filename,
                    image_path=db_record.image_path,
                    recognition_result=json.loads(db_record.recognition_result),
                    timestamp=db_record.created_at,
                    model_used=db_record.model_used,
                    processing_time=db_record.processing_time,
                    is_multi_role=db_record.is_multi_role,
                    nsfw_status=db_record.nsfw_status,
                    detected_text=db_record.detected_text,
                )
                logger.info(f"数据库创建识别记录: {record_id}")
                return new_record
            except Exception as e:
                logger.error(f"数据库创建识别记录失败: {e}")

        new_record = RecognitionRecord(
            id=record_id,
            user_id=record.user_id,
            username=record.username,
            image_filename=record.image_filename,
            image_path=record.image_path,
            recognition_result=record.recognition_result,
            model_used=record.model_used,
            processing_time=record.processing_time,
            is_multi_role=record.is_multi_role,
            nsfw_status=record.nsfw_status,
            detected_text=record.detected_text,
        )

        self.records.append(new_record)
        self._save_records()
        logger.info(f"创建识别记录: {record_id}")
        return new_record

    def get_records_by_user(self, user_id: str) -> List[RecognitionRecord]:
        """根据用户ID获取识别记录"""
        if _use_database:
            try:
                from src.services.support.database_service import RecognitionRecordDB, get_db_service

                db = get_db_service()
                # 先查用户自己的记录
                db_records = RecognitionRecordDB.get_by_user(db, user_id)
                # 如果为空，再查匿名记录
                if not db_records and user_id != "anonymous":
                    db_records = RecognitionRecordDB.get_by_user(db, "anonymous")
                return [
                    RecognitionRecord(
                        id=r.id,
                        user_id=r.user_id,
                        username=r.username,
                        image_filename=r.image_filename,
                        image_path=r.image_path,
                        recognition_result=json.loads(r.recognition_result),
                        timestamp=r.created_at,
                        model_used=r.model_used,
                        processing_time=r.processing_time,
                        is_multi_role=r.is_multi_role,
                        nsfw_status=r.nsfw_status,
                        detected_text=r.detected_text,
                    )
                    for r in db_records
                ]
            except Exception as e:
                logger.error(f"数据库获取用户记录失败: {e}")

        user_records = [r for r in self.records if r.user_id == user_id]
        # 如果为空，再查匿名记录
        if not user_records and user_id != "anonymous":
            user_records = [r for r in self.records if r.user_id == "anonymous"]
        user_records.sort(key=lambda x: x.timestamp, reverse=True)
        return user_records

    def get_record_by_id(self, record_id: str) -> Optional[RecognitionRecord]:
        """根据记录ID获取识别记录"""
        if _use_database:
            try:
                from src.services.support.database_service import RecognitionRecordDB, get_db_service

                db = get_db_service()
                db_record = RecognitionRecordDB.get_by_id(db, record_id)
                if db_record:
                    return RecognitionRecord(
                        id=db_record.id,
                        user_id=db_record.user_id,
                        username=db_record.username,
                        image_filename=db_record.image_filename,
                        image_path=db_record.image_path,
                        recognition_result=json.loads(db_record.recognition_result),
                        timestamp=db_record.created_at,
                        model_used=db_record.model_used,
                        processing_time=db_record.processing_time,
                        is_multi_role=db_record.is_multi_role,
                        nsfw_status=db_record.nsfw_status,
                        detected_text=db_record.detected_text,
                    )
                return None
            except Exception as e:
                logger.error(f"数据库获取记录失败: {e}")

        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def delete_record(self, record_id: str) -> bool:
        """删除识别记录"""
        if _use_database:
            try:
                from src.services.support.database_service import RecognitionRecordDB, get_db_service

                db = get_db_service()
                return RecognitionRecordDB.delete(db, record_id)
            except Exception as e:
                logger.error(f"数据库删除记录失败: {e}")

        original_length = len(self.records)
        self.records = [r for r in self.records if r.id != record_id]
        if len(self.records) < original_length:
            self._save_records()
            logger.info(f"删除识别记录: {record_id}")
            return True
        return False

    def get_all_records(self) -> List[RecognitionRecord]:
        """获取所有识别记录"""
        if _use_database:
            try:
                from src.services.support.database_service import RecognitionRecordDB, get_db_service

                db = get_db_service()
                db_records = RecognitionRecordDB.get_all(db)
                return [
                    RecognitionRecord(
                        id=r.id,
                        user_id=r.user_id,
                        username=r.username,
                        image_filename=r.image_filename,
                        image_path=r.image_path,
                        recognition_result=json.loads(r.recognition_result),
                        timestamp=r.created_at,
                        model_used=r.model_used,
                        processing_time=r.processing_time,
                        is_multi_role=r.is_multi_role,
                        nsfw_status=r.nsfw_status,
                        detected_text=r.detected_text,
                    )
                    for r in db_records
                ]
            except Exception as e:
                logger.error(f"数据库获取所有记录失败: {e}")

        sorted_records = sorted(self.records, key=lambda x: x.timestamp, reverse=True)
        return sorted_records


recognition_service = RecognitionService()


def get_recognition_service() -> RecognitionService:
    """获取识别记录服务实例"""
    return recognition_service