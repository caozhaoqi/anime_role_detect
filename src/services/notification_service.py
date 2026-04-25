#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消息推送服务
支持企业微信、飞书等平台的消息推送
"""

import os
import json
import time
from typing import Optional, Dict, Any, List
from enum import Enum
import requests
from src.core.logging.global_logger import get_logger

logger = get_logger("notification_service")


class NotificationPlatform(Enum):
    """支持的推送平台"""
    WECOM = "wecom"           # 企业微信
    FEISHU = "feishu"         # 飞书
    WXPUSHER = "wxpusher"     # 微信推送
    DingTalk = "dingtalk"     # 钉钉


class NotificationManager:
    """消息通知管理器"""

    _instance: Optional['NotificationManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.enabled = os.environ.get('NOTIFICATION_ENABLED', 'false').lower() == 'true'
        self.platform = os.environ.get('NOTIFICATION_PLATFORM', 'wecom').lower()

        self.wecom_webhook = os.environ.get('WECOM_WEBHOOK_URL', '')
        self.wecom_corp_id = os.environ.get('WECOM_CORP_ID', '')
        self.wecom_agent_id = os.environ.get('WECOM_AGENT_ID', '')
        self.wecom_secret = os.environ.get('WECOM_SECRET', '')

        self.feishu_webhook = os.environ.get('FEISHU_WEBHOOK_URL', '')
        self.feishu_app_id = os.environ.get('FEISHU_APP_ID', '')
        self.feishu_app_secret = os.environ.get('FEISHU_APP_SECRET', '')
        self.feishu_receive_id = os.environ.get('FEISHU_RECEIVE_ID', '')
        self.feishu_receive_id_type = os.environ.get('FEISHU_RECEIVE_ID_TYPE', 'chat_id')

        self.wxpusher_token = os.environ.get('WXPUSHER_TOKEN', '')
        self.wxpusher_uids = os.environ.get('WXPUSHER_UIDS', '').split(',')

        self.dingtalk_webhook = os.environ.get('DINGTALK_WEBHOOK_URL', '')

        self._wecom_access_token = None
        self._wecom_token_expires = 0
        self._feishu_access_token = None
        self._feishu_token_expires = 0

        logger.info(f"通知服务初始化完成，平台: {self.platform}, 启用: {self.enabled}")

    def _get_wecom_access_token(self) -> Optional[str]:
        """获取企业微信Access Token"""
        if self._wecom_access_token and time.time() < self._wecom_token_expires:
            return self._wecom_access_token

        try:
            url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
            params = {
                "corpid": self.wecom_corp_id,
                "corpsecret": self.wecom_secret
            }
            response = requests.get(url, params=params, timeout=10)
            result = response.json()

            if result.get("errcode") == 0:
                self._wecom_access_token = result["access_token"]
                self._wecom_token_expires = time.time() + result.get("expires_in", 7200) - 300
                return self._wecom_access_token
            else:
                logger.error(f"获取企业微信Access Token失败: {result}")
                return None
        except Exception as e:
            logger.error(f"获取企业微信Access Token异常: {e}")
            return None

    def _get_feishu_access_token(self) -> Optional[str]:
        """获取飞书Access Token"""
        if self._feishu_access_token and time.time() < self._feishu_token_expires:
            return self._feishu_access_token

        try:
            url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            headers = {"Content-Type": "application/json"}
            data = {
                "app_id": self.feishu_app_id,
                "app_secret": self.feishu_app_secret
            }
            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()

            if result.get("code") == 0:
                self._feishu_access_token = result["tenant_access_token"]
                self._feishu_token_expires = time.time() + result.get("expire", 7200) - 300
                return self._feishu_access_token
            else:
                logger.error(f"获取飞书Access Token失败: {result}")
                return None
        except Exception as e:
            logger.error(f"获取飞书Access Token异常: {e}")
            return None

    def send_wecom_message(self, content: str, mentioned_list: List[str] = None) -> bool:
        """发送企业微信消息"""
        try:
            access_token = self._get_wecom_access_token()
            if not access_token:
                return False

            url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"
            data = {
                "touser": "|".join(mentioned_list) if mentioned_list else "@all",
                "msgtype": "text",
                "agentid": self.wecom_agent_id,
                "text": {
                    "content": content
                }
            }

            response = requests.post(url, json=data, timeout=10)
            result = response.json()

            if result.get("errcode") == 0:
                logger.info("企业微信消息发送成功")
                return True
            else:
                logger.error(f"企业微信消息发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"企业微信消息发送异常: {e}")
            return False

    def send_wecom_markdown(self, content: str) -> bool:
        """发送企业微信Markdown消息"""
        try:
            access_token = self._get_wecom_access_token()
            if not access_token:
                return False

            url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"
            data = {
                "touser": "@all",
                "msgtype": "markdown",
                "agentid": self.wecom_agent_id,
                "markdown": {
                    "content": content
                }
            }

            response = requests.post(url, json=data, timeout=10)
            result = response.json()

            if result.get("errcode") == 0:
                logger.info("企业微信Markdown消息发送成功")
                return True
            else:
                logger.error(f"企业微信Markdown消息发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"企业微信Markdown消息发送异常: {e}")
            return False

    def send_feishu_message(self, content: str, msg_type: str = "text") -> bool:
        """发送飞书消息"""
        try:
            # 优先使用webhook方式
            if self.feishu_webhook:
                url = self.feishu_webhook
                headers = {"Content-Type": "application/json"}
                
                # 飞书机器人webhook格式
                data = {
                    "msg_type": "text",
                    "content": {
                        "text": content
                    }
                }
                
                response = requests.post(url, headers=headers, json=data, timeout=10)
                result = response.json()
                
                if result.get("code") == 0:
                    logger.info("飞书webhook消息发送成功")
                    return True
                else:
                    logger.error(f"飞书webhook消息发送失败: {result}")
                    # 如果webhook失败，尝试使用API方式
            
            # API方式
            access_token = self._get_feishu_access_token()
            if not access_token:
                return False

            url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            if msg_type == "text":
                data = {
                    "receive_id": self.feishu_receive_id or "ou_xxxxxx",
                    "msg_type": "text",
                    "content": json.dumps({"text": content})
                }
            else:
                data = {
                    "receive_id": self.feishu_receive_id or "ou_xxxxxx",
                    "msg_type": "post",
                    "content": json.dumps({
                        "title": "训练通知",
                        "content": [[{"tag": "text", "text": content}]]
                    })
                }

            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()

            if result.get("code") == 0:
                logger.info("飞书API消息发送成功")
                return True
            else:
                logger.error(f"飞书API消息发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"飞书消息发送异常: {e}")
            return False

    def send_feishu_card(self, card_content: Dict[str, Any]) -> bool:
        """发送飞书卡片消息"""
        try:
            access_token = self._get_feishu_access_token()
            if not access_token:
                return False

            url = "https://open.feishu.cn/open-apis/im/v1/messages"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            data = {
                "receive_id": self.feishu_receive_id or "ou_xxxxxx",
                "receive_id_type": self.feishu_receive_id_type,
                "msg_type": "interactive",
                "content": json.dumps(card_content)
            }

            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()

            if result.get("code") == 0:
                logger.info("飞书卡片消息发送成功")
                return True
            else:
                logger.error(f"飞书卡片消息发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"飞书卡片消息发送异常: {e}")
            return False

    def send_wxpusher_message(self, content: str, summary: str = "消息摘要") -> bool:
        """发送WxPusher消息"""
        try:
            if not self.wxpusher_token:
                logger.warning("WxPusher token未配置")
                return False

            url = "https://wxpusher.zjiecode.com/api/send/message"
            data = {
                "appToken": self.wxpusher_token,
                "content": content,
                "summary": summary,
                "contentType": 1,
                "uids": self.wxpusher_uids if self.wxpusher_uids else [],
                "verifyPay": False
            }

            response = requests.post(url, json=data, timeout=10)
            result = response.json()

            if result.get("code") == 1000:
                logger.info("WxPusher消息发送成功")
                return True
            else:
                logger.error(f"WxPusher消息发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"WxPusher消息发送异常: {e}")
            return False

    def send_dingtalk_message(self, content: str, msg_type: str = "text") -> bool:
        """发送钉钉消息"""
        try:
            if not self.dingtalk_webhook:
                logger.warning("钉钉webhook未配置")
                return False

            url = self.dingtalk_webhook
            data = {
                "msgtype": msg_type,
                msg_type: {
                    "content": content
                }
            }

            response = requests.post(url, json=data, timeout=10)
            result = response.json()

            if result.get("errcode") == 0:
                logger.info("钉钉消息发送成功")
                return True
            else:
                logger.error(f"钉钉消息发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"钉钉消息发送异常: {e}")
            return False

    def send(self, message: str, title: str = None, level: str = "info") -> bool:
        """
        发送通知消息（统一接口）

        Args:
            message: 消息内容
            title: 消息标题
            level: 消息级别 (info, warning, error, success)

        Returns:
            bool: 是否发送成功
        """
        if not self.enabled:
            logger.debug("通知功能未启用，跳过发送")
            return False

        level_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }
        emoji = level_emoji.get(level, "ℹ️")

        formatted_message = f"{emoji} {message}" if message else f"{emoji} {title}"

        if self.platform == "wecom":
            return self.send_wecom_message(formatted_message)
        elif self.platform == "feishu":
            return self.send_feishu_message(formatted_message)
        elif self.platform == "wxpusher":
            return self.send_wxpusher_message(formatted_message, summary=title or "训练通知")
        elif self.platform == "dingtalk":
            return self.send_dingtalk_message(formatted_message)
        else:
            logger.warning(f"未知的推送平台: {self.platform}")
            return False

    def send_training_progress(
        self,
        stage: str,
        progress: float,
        message: str,
        metrics: Dict[str, Any] = None
    ) -> bool:
        """
        发送训练进度通知

        Args:
            stage: 当前阶段 (数据采集中, 训练中, 评估中, 部署完成)
            progress: 进度百分比 (0-100)
            message: 详细消息
            metrics: 训练指标

        Returns:
            bool: 是否发送成功
        """
        if not self.enabled:
            return False

        progress_bar = "█" * int(progress / 10) + "░" * (10 - int(progress / 10))

        title = f"【训练进度】{stage}"
        content = f"""
{message}
进度: {progress_bar} {progress:.1f}%
        """

        if metrics:
            metric_lines = []
            for k, v in metrics.items():
                metric_lines.append(f"- {k}: {v}")
            content += "\n" + "\n".join(metric_lines)

        return self.send(content, title=title, level="info")

    def send_training_complete(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        model_path: str = None,
        training_time: float = None
    ) -> bool:
        """
        发送训练完成通知

        Args:
            model_name: 模型名称
            metrics: 训练指标
            model_path: 模型路径
            training_time: 训练耗时（秒）

        Returns:
            bool: 是否发送成功
        """
        if not self.enabled:
            return False

        time_str = f"{training_time / 3600:.1f}小时" if training_time and training_time >= 3600 else f"{training_time / 60:.1f}分钟" if training_time else "未知"

        content = f"""✅ 模型训练完成

模型: {model_name}
训练耗时: {time_str}

📊 评估指标:"""

        for k, v in metrics.items():
            if isinstance(v, float):
                content += f"\n- {k}: {v:.4f}"
            else:
                content += f"\n- {k}: {v}"

        if model_path:
            content += f"\n\n📁 模型路径: {model_path}"

        return self.send(content, title=f"训练完成 - {model_name}", level="success")

    def send_training_error(
        self,
        stage: str,
        error_message: str
    ) -> bool:
        """
        发送训练错误通知

        Args:
            stage: 出错的阶段
            error_message: 错误信息

        Returns:
            bool: 是否发送成功
        """
        if not self.enabled:
            return False

        content = f"""❌ 训练出错

阶段: {stage}
错误: {error_message}
        """

        return self.send(content, title=f"训练错误 - {stage}", level="error")


_notification_manager = None


def get_notification_manager() -> NotificationManager:
    """获取通知管理器实例"""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


def send_notification(message: str, title: str = None, level: str = "info") -> bool:
    """发送通知（快捷函数）"""
    return get_notification_manager().send(message, title, level)


def send_training_progress_notification(
    stage: str,
    progress: float,
    message: str,
    metrics: Dict[str, Any] = None
) -> bool:
    """发送训练进度通知（快捷函数）"""
    return get_notification_manager().send_training_progress(stage, progress, message, metrics)


def send_training_complete_notification(
    model_name: str,
    metrics: Dict[str, Any],
    model_path: str = None,
    training_time: float = None
) -> bool:
    """发送训练完成通知（快捷函数）"""
    return get_notification_manager().send_training_complete(model_name, metrics, model_path, training_time)


def send_training_error_notification(stage: str, error_message: str) -> bool:
    """发送训练错误通知（快捷函数）"""
    return get_notification_manager().send_training_error(stage, error_message)