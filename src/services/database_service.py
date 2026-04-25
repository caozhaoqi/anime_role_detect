#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库服务
提供数据库操作的封装
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from src.core.config.database import get_db, init_database, create_tables
from src.core.logging.global_logger import get_logger
from src.models.database_models import RecognitionRecordModel, UserFeedbackModel

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
        detected_text: bool
    ) -> RecognitionRecordModel:
        """创建识别记录"""
        record = RecognitionRecordModel(
            id=record_id,
            user_id=user_id,
            username=username,
            image_filename=image_filename,
            image_path=image_path,
            recognition_result=recognition_result,
            model_used=model_used,
            processing_time=processing_time,
            is_multi_role=is_multi_role,
            nsfw_status=nsfw_status,
            detected_text=detected_text,
            timestamp=datetime.now()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(f"数据库创建识别记录: {record_id}")
        return record

    @staticmethod
    def get_by_id(db: Session, record_id: str) -> Optional[RecognitionRecordModel]:
        """根据ID获取识别记录"""
        return db.query(RecognitionRecordModel).filter(
            RecognitionRecordModel.id == record_id
        ).first()

    @staticmethod
    def get_by_user(
        db: Session,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[RecognitionRecordModel]:
        """获取用户的所有识别记录"""
        return db.query(RecognitionRecordModel).filter(
            RecognitionRecordModel.user_id == user_id
        ).order_by(
            desc(RecognitionRecordModel.timestamp)
        ).offset(offset).limit(limit).all()

    @staticmethod
    def get_all(
        db: Session,
        limit: int = 100,
        offset: int = 0
    ) -> List[RecognitionRecordModel]:
        """获取所有识别记录"""
        return db.query(RecognitionRecordModel).order_by(
            desc(RecognitionRecordModel.timestamp)
        ).offset(offset).limit(limit).all()

    @staticmethod
    def count_by_user(db: Session, user_id: str) -> int:
        """统计用户的识别记录数量"""
        return db.query(RecognitionRecordModel).filter(
            RecognitionRecordModel.user_id == user_id
        ).count()

    @staticmethod
    def delete(db: Session, record_id: str) -> bool:
        """删除识别记录"""
        record = db.query(RecognitionRecordModel).filter(
            RecognitionRecordModel.id == record_id
        ).first()
        if record:
            db.delete(record)
            db.commit()
            logger.info(f"数据库删除识别记录: {record_id}")
            return True
        return False

    @staticmethod
    def delete_by_user(db: Session, user_id: str) -> int:
        """删除用户的所有识别记录"""
        records = db.query(RecognitionRecordModel).filter(
            RecognitionRecordModel.user_id == user_id
        ).all()
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
        extra_data: Dict[str, Any] = None
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
            extra_data=extra_data or {},
            created_at=datetime.now()
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        logger.info(f"数据库创建用户反馈: {feedback_id}")
        return feedback

    @staticmethod
    def get_by_id(db: Session, feedback_id: str) -> Optional[UserFeedbackModel]:
        """根据ID获取用户反馈"""
        return db.query(UserFeedbackModel).filter(
            UserFeedbackModel.id == feedback_id
        ).first()

    @staticmethod
    def get_by_user(
        db: Session,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserFeedbackModel]:
        """获取用户的所有反馈"""
        return db.query(UserFeedbackModel).filter(
            UserFeedbackModel.user_id == user_id
        ).order_by(
            desc(UserFeedbackModel.created_at)
        ).offset(offset).limit(limit).all()

    @staticmethod
    def get_by_record(
        db: Session,
        record_id: str
    ) -> List[UserFeedbackModel]:
        """获取特定记录的所有反馈"""
        return db.query(UserFeedbackModel).filter(
            UserFeedbackModel.record_id == record_id
        ).order_by(
            desc(UserFeedbackModel.created_at)
        ).all()

    @staticmethod
    def delete(db: Session, feedback_id: str) -> bool:
        """删除用户反馈"""
        feedback = db.query(UserFeedbackModel).filter(
            UserFeedbackModel.id == feedback_id
        ).first()
        if feedback:
            db.delete(feedback)
            db.commit()
            logger.info(f"数据库删除用户反馈: {feedback_id}")
            return True
        return False