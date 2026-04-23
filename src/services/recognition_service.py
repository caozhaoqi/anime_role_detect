#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别记录服务

管理用户的识别历史记录
"""

import json
import os
from datetime import datetime
from typing import List, Optional
from src.models.recognition_record import RecognitionRecord, RecognitionRecordCreate
from src.core.logging.global_logger import get_logger

logger = get_logger("recognition_service")


class RecognitionService:
    """识别记录服务"""
    
    def __init__(self):
        """初始化识别记录服务"""
        self.records_file = "recognition_records.json"
        self.records = self._load_records()
    
    def _load_records(self) -> List[RecognitionRecord]:
        """
        从文件加载识别记录
        
        Returns:
            List[RecognitionRecord]: 识别记录列表
        """
        try:
            if os.path.exists(self.records_file):
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    records = []
                    for item in data:
                        # 转换时间字符串为datetime对象
                        if 'timestamp' in item:
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        records.append(RecognitionRecord(**item))
                    logger.info(f"加载了 {len(records)} 条识别记录")
                    return records
            return []
        except Exception as e:
            logger.error(f"加载识别记录失败: {e}")
            return []
    
    def _save_records(self):
        """
        保存识别记录到文件
        """
        try:
            data = []
            for record in self.records:
                record_dict = record.model_dump()
                # 转换datetime对象为字符串
                if 'timestamp' in record_dict and isinstance(record_dict['timestamp'], datetime):
                    record_dict['timestamp'] = record_dict['timestamp'].isoformat()
                data.append(record_dict)
            
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"保存了 {len(self.records)} 条识别记录")
        except Exception as e:
            logger.error(f"保存识别记录失败: {e}")
    
    def create_record(self, record: RecognitionRecordCreate) -> RecognitionRecord:
        """
        创建新的识别记录
        
        Args:
            record: 识别记录创建对象
        
        Returns:
            RecognitionRecord: 创建的识别记录
        """
        try:
            # 生成唯一ID
            record_id = f"rec_{int(datetime.now().timestamp() * 1000)}"
            
            # 创建识别记录
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
                detected_text=record.detected_text
            )
            
            # 添加到记录列表
            self.records.append(new_record)
            
            # 保存到文件
            self._save_records()
            
            logger.info(f"创建了新的识别记录: {record_id}")
            return new_record
        except Exception as e:
            logger.error(f"创建识别记录失败: {e}")
            raise
    
    def get_records_by_user(self, user_id: str) -> List[RecognitionRecord]:
        """
        根据用户ID获取识别记录
        
        Args:
            user_id: 用户ID
        
        Returns:
            List[RecognitionRecord]: 用户的识别记录列表
        """
        try:
            user_records = [record for record in self.records if record.user_id == user_id]
            # 按时间倒序排序
            user_records.sort(key=lambda x: x.timestamp, reverse=True)
            logger.info(f"获取用户 {user_id} 的 {len(user_records)} 条识别记录")
            return user_records
        except Exception as e:
            logger.error(f"获取用户识别记录失败: {e}")
            return []
    
    def get_record_by_id(self, record_id: str) -> Optional[RecognitionRecord]:
        """
        根据记录ID获取识别记录
        
        Args:
            record_id: 记录ID
        
        Returns:
            Optional[RecognitionRecord]: 识别记录
        """
        try:
            for record in self.records:
                if record.id == record_id:
                    return record
            return None
        except Exception as e:
            logger.error(f"获取识别记录失败: {e}")
            return None
    
    def delete_record(self, record_id: str) -> bool:
        """
        删除识别记录
        
        Args:
            record_id: 记录ID
        
        Returns:
            bool: 删除是否成功
        """
        try:
            original_length = len(self.records)
            self.records = [record for record in self.records if record.id != record_id]
            if len(self.records) < original_length:
                self._save_records()
                logger.info(f"删除了识别记录: {record_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除识别记录失败: {e}")
            return False
    
    def get_all_records(self) -> List[RecognitionRecord]:
        """
        获取所有识别记录
        
        Returns:
            List[RecognitionRecord]: 所有识别记录
        """
        try:
            # 按时间倒序排序
            sorted_records = sorted(self.records, key=lambda x: x.timestamp, reverse=True)
            return sorted_records
        except Exception as e:
            logger.error(f"获取所有识别记录失败: {e}")
            return []


# 全局识别记录服务实例
recognition_service = RecognitionService()


def get_recognition_service() -> RecognitionService:
    """
    获取识别记录服务实例
    
    Returns:
        RecognitionService: 识别记录服务实例
    """
    return recognition_service
