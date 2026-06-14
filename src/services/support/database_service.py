#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库服务
提供数据库操作的封装
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from src.core.config.database import get_db, init_database, create_tables
from src.core.logging.global_logger import get_logger
from src.models.database_models import RecognitionRecordModel, UserFeedbackModel, CleaningRecordModel

logger = get_logger("db_service")

_db_session = None


def get_db_service() -> Session:
    """获取数据库会话"""
    global _db_session
    if _db_session is None:
        init_database()
        create_tables()
        from src.core.config.database import SessionLocal

        _db_session = SessionLocal()
    return _db_session


def close_db_service():
    """关闭数据库会话"""
    global _db_session
    if _db_session:
        _db_session.close()
        _db_session = None


class RecognitionRecordDB:
    """识别记录数据库操作类"""

    @staticmethod
    def create(
        db: Session,
        record_id: str,
        user_id: str,
        username: str,
        image_filename: str,
        image_path: str,
        recognition_result: Dict[str, Any],
        model_used: str,
        processing_time: float,
        is_multi_role: bool,
        nsfw_status: bool,
        detected_text: bool,
    ) -> RecognitionRecordModel:
        """创建识别记录"""
        record = RecognitionRecordModel(
            id=record_id,
            user_id=user_id,
            username=username,
            image_filename=image_filename,
            image_path=image_path,
            recognition_result=json.dumps(recognition_result, ensure_ascii=False),
            model_used=model_used,
            processing_time=int(processing_time),
            is_multi_role=is_multi_role,
            nsfw_status=nsfw_status,
            detected_text=detected_text,
            created_at=datetime.now(),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(f"数据库创建识别记录: {record_id}")
        return record

    @staticmethod
    def get_by_id(db: Session, record_id: str) -> Optional[RecognitionRecordModel]:
        """根据ID获取识别记录"""
        return (
            db.query(RecognitionRecordModel).filter(RecognitionRecordModel.id == record_id).first()
        )

    @staticmethod
    def get_by_user(
        db: Session, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[RecognitionRecordModel]:
        """获取用户的所有识别记录"""
        return (
            db.query(RecognitionRecordModel)
            .filter(RecognitionRecordModel.user_id == user_id)
            .order_by(desc(RecognitionRecordModel.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_all(db: Session, limit: int = 100, offset: int = 0) -> List[RecognitionRecordModel]:
        """获取所有识别记录"""
        return (
            db.query(RecognitionRecordModel)
            .order_by(desc(RecognitionRecordModel.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def count_by_user(db: Session, user_id: str) -> int:
        """统计用户的识别记录数量"""
        return (
            db.query(RecognitionRecordModel)
            .filter(RecognitionRecordModel.user_id == user_id)
            .count()
        )

    @staticmethod
    def delete(db: Session, record_id: str) -> bool:
        """删除识别记录"""
        record = (
            db.query(RecognitionRecordModel).filter(RecognitionRecordModel.id == record_id).first()
        )
        if record:
            db.delete(record)
            db.commit()
            logger.info(f"数据库删除识别记录: {record_id}")
            return True
        return False

    @staticmethod
    def delete_by_user(db: Session, user_id: str) -> int:
        """删除用户的所有识别记录"""
        records = (
            db.query(RecognitionRecordModel).filter(RecognitionRecordModel.user_id == user_id).all()
        )
        count = len(records)
        for record in records:
            db.delete(record)
        db.commit()
        logger.info(f"数据库删除用户 {user_id} 的 {count} 条识别记录")
        return count


class UserFeedbackDB:
    """用户反馈数据库操作类"""

    @staticmethod
    def create(
        db: Session,
        feedback_id: str,
        user_id: str,
        feedback_type: str,
        username: str = None,
        record_id: str = None,
        image_filename: str = None,
        rating: float = None,
        comment: str = None,
        correction_role: str = None,
        is_correct: bool = None,
        extra_data: Dict[str, Any] = None,
    ) -> UserFeedbackModel:
        """创建用户反馈"""
        feedback = UserFeedbackModel(
            id=feedback_id,
            user_id=user_id,
            username=username,
            record_id=record_id,
            image_filename=image_filename,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            correction_role=correction_role,
            is_correct=is_correct,
            extra_data=json.dumps(extra_data or {}, ensure_ascii=False),
            created_at=datetime.now(),
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        logger.info(f"数据库创建用户反馈: {feedback_id}")
        return feedback

    @staticmethod
    def get_by_id(db: Session, feedback_id: str) -> Optional[UserFeedbackModel]:
        """根据ID获取用户反馈"""
        return db.query(UserFeedbackModel).filter(UserFeedbackModel.id == feedback_id).first()

    @staticmethod
    def get_by_user(
        db: Session, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[UserFeedbackModel]:
        """获取用户的所有反馈"""
        return (
            db.query(UserFeedbackModel)
            .filter(UserFeedbackModel.user_id == user_id)
            .order_by(desc(UserFeedbackModel.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_by_record(db: Session, record_id: str) -> List[UserFeedbackModel]:
        """获取特定记录的所有反馈"""
        return (
            db.query(UserFeedbackModel)
            .filter(UserFeedbackModel.record_id == record_id)
            .order_by(desc(UserFeedbackModel.created_at))
            .all()
        )

    @staticmethod
    def delete(db: Session, feedback_id: str) -> bool:
        """删除用户反馈"""
        feedback = db.query(UserFeedbackModel).filter(UserFeedbackModel.id == feedback_id).first()
        if feedback:
            db.delete(feedback)
            db.commit()
            logger.info(f"数据库删除用户反馈: {feedback_id}")
            return True
        return False


class CleaningRecordDB:
    """数据清洗记录数据库操作类"""

    @staticmethod
    def create(
        db: Session,
        record_id: str,
        user_id: str,
        username: str,
        input_dir: str = None,
        output_dir: str = None,
        config: Dict[str, Any] = None,
    ) -> CleaningRecordModel:
        """创建数据清洗记录"""
        record = CleaningRecordModel(
            id=record_id,
            user_id=user_id,
            username=username,
            input_dir=input_dir,
            output_dir=output_dir,
            config=json.dumps(config or {}, ensure_ascii=False),
            status="pending",
            created_at=datetime.now(),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(f"数据库创建数据清洗记录: {record_id}")
        return record

    @staticmethod
    def get_by_id(db: Session, record_id: str) -> Optional[CleaningRecordModel]:
        """根据ID获取数据清洗记录"""
        return db.query(CleaningRecordModel).filter(CleaningRecordModel.id == record_id).first()

    @staticmethod
    def get_by_user(
        db: Session, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[CleaningRecordModel]:
        """获取用户的所有数据清洗记录"""
        return (
            db.query(CleaningRecordModel)
            .filter(CleaningRecordModel.user_id == user_id)
            .order_by(desc(CleaningRecordModel.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_all(db: Session, limit: int = 50, offset: int = 0) -> List[CleaningRecordModel]:
        """获取所有数据清洗记录"""
        return (
            db.query(CleaningRecordModel)
            .order_by(desc(CleaningRecordModel.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update_status(
        db: Session, record_id: str, status: str, **kwargs
    ) -> Optional[CleaningRecordModel]:
        """更新数据清洗记录状态"""
        record = db.query(CleaningRecordModel).filter(CleaningRecordModel.id == record_id).first()
        if record:
            record.status = status
            if "started_at" in kwargs:
                record.started_at = kwargs["started_at"]
            if "completed_at" in kwargs:
                record.completed_at = kwargs["completed_at"]
            if "total_files" in kwargs:
                record.total_files = kwargs["total_files"]
            if "processed_files" in kwargs:
                record.processed_files = kwargs["processed_files"]
            if "valid_files" in kwargs:
                record.valid_files = kwargs["valid_files"]
            if "rejected_files" in kwargs:
                record.rejected_files = kwargs["rejected_files"]
            if "duplicate_files" in kwargs:
                record.duplicate_files = kwargs["duplicate_files"]
            if "report_path" in kwargs:
                record.report_path = kwargs["report_path"]
            if "error_message" in kwargs:
                record.error_message = kwargs["error_message"]
            if "duration_seconds" in kwargs:
                record.duration_seconds = kwargs["duration_seconds"]
            db.commit()
            db.refresh(record)
            logger.info(f"数据库更新数据清洗记录状态: {record_id} -> {status}")
            return record
        return None

    @staticmethod
    def count_by_user(db: Session, user_id: str) -> int:
        """统计用户的数据清洗记录数量"""
        return (
            db.query(CleaningRecordModel)
            .filter(CleaningRecordModel.user_id == user_id)
            .count()
        )

    @staticmethod
    def delete(db: Session, record_id: str) -> bool:
        """删除数据清洗记录"""
        record = db.query(CleaningRecordModel).filter(CleaningRecordModel.id == record_id).first()
        if record:
            db.delete(record)
            db.commit()
            logger.info(f"数据库删除数据清洗记录: {record_id}")
            return True
        return False

    @staticmethod
    def delete_by_user(db: Session, user_id: str) -> int:
        """删除用户的所有数据清洗记录"""
        records = (
            db.query(CleaningRecordModel).filter(CleaningRecordModel.user_id == user_id).all()
        )
        count = len(records)
        for record in records:
            db.delete(record)
        db.commit()
        logger.info(f"数据库删除用户 {user_id} 的 {count} 条数据清洗记录")
        return count