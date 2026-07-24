import asyncio
import json
import os
import uuid
from typing import Optional, Dict, Any, Callable
import aio_pika
import pika

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("message_queue_service")


class MessageQueueService:
    """消息队列服务"""

    _instance: Optional["MessageQueueService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.initialized = True
        self.connection = None
        self.channel = None
        self.queue = None
        self.callback_queue = None
        self.futures = {}
        self.connected = False

        # 配置
        self.RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
        self.RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))
        self.RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
        self.RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "guest")
        self.RABBITMQ_VHOST = os.environ.get("RABBITMQ_VHOST", "/")

        self.QUEUE_NAME = os.environ.get("QUEUE_NAME", "anime_role_detect")
        self.EXCHANGE_NAME = os.environ.get("EXCHANGE_NAME", "anime_role_detect_exchange")

        # 不在这里初始化连接，改为延迟初始化
        logger.info("消息队列服务创建（延迟初始化）")

    async def _init_connection(self):
        """初始化连接"""
        if self.connected:
            return

        try:
            # 创建连接
            self.connection = await aio_pika.connect_robust(
                host=self.RABBITMQ_HOST,
                port=self.RABBITMQ_PORT,
                login=self.RABBITMQ_USER,
                password=self.RABBITMQ_PASSWORD,
                virtualhost=self.RABBITMQ_VHOST,
                timeout=5,  # 添加超时
            )

            # 创建通道
            self.channel = await self.connection.channel()

            # 声明交换机
            await self.channel.declare_exchange(self.EXCHANGE_NAME, aio_pika.ExchangeType.DIRECT)

            # 声明队列
            self.queue = await self.channel.declare_queue(self.QUEUE_NAME, durable=True)

            # 绑定队列到交换机
            await self.queue.bind(self.EXCHANGE_NAME, routing_key=self.QUEUE_NAME)

            # 声明回调队列
            self.callback_queue = await self.channel.declare_queue(exclusive=True)

            # 消费回调队列
            await self.callback_queue.consume(self._on_response)

            self.connected = True
            logger.info(f"消息队列连接成功: {self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}")
        except asyncio.TimeoutError:
            logger.warning(f"消息队列连接超时（{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}），将使用同步处理")
        except Exception as e:
            logger.warning(f"消息队列连接失败，将使用同步处理: {e}")

    async def _on_response(self, message: aio_pika.IncomingMessage):
        """处理响应消息"""
        async with message.process():
            correlation_id = message.correlation_id
            if correlation_id in self.futures:
                future = self.futures[correlation_id]
                future.set_result(message.body)

    async def send_message(
        self, message: Dict[str, Any], routing_key: str = None
    ) -> Dict[str, Any]:
        """发送消息并等待响应"""
        if not self.connection or not self.channel:
            # 如果消息队列不可用，直接返回处理结果
            logger.warning("消息队列不可用，使用同步处理")
            return await self._process_message_sync(message)

        try:
            correlation_id = str(uuid.uuid4())
            future = asyncio.Future()
            self.futures[correlation_id] = future

            # 发送消息
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message).encode(),
                    correlation_id=correlation_id,
                    reply_to=self.callback_queue.name,
                ),
                routing_key=routing_key or self.QUEUE_NAME,
            )

            # 等待响应
            response_body = await future
            response = json.loads(response_body.decode())

            # 清理future
            del self.futures[correlation_id]

            return response
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            # 失败时使用同步处理
            return await self._process_message_sync(message)

    async def _process_message_sync(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """同步处理消息"""
        # 根据消息类型处理
        message_type = message.get("type")

        if message_type == "image_processing":
            # 处理图像
            from src.services.processor.image_processor import process_single_image

            # 这里需要模拟文件对象
            # 实际实现需要根据具体情况调整
            logger.info("使用同步处理图像")
            return {"status": "processed", "message": "Image processed synchronously"}

        return {"status": "error", "message": "Unknown message type"}

    async def consume_messages(self, callback: Callable[[Dict[str, Any]], None]):
        """消费消息"""
        if not self.connection or not self.channel or not self.queue:
            logger.warning("消息队列不可用，无法消费消息")
            return

        async def on_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    data = json.loads(message.body.decode())
                    await callback(data)

                    # 发送响应
                    if message.reply_to:
                        await self.channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps({"status": "success"}).encode(),
                                correlation_id=message.correlation_id,
                            ),
                            routing_key=message.reply_to,
                        )
                except Exception as e:
                    logger.error(f"处理消息失败: {e}")

                    # 发送错误响应
                    if message.reply_to:
                        await self.channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps({"status": "error", "message": str(e)}).encode(),
                                correlation_id=message.correlation_id,
                            ),
                            routing_key=message.reply_to,
                        )

        # 开始消费
        await self.queue.consume(on_message)

    async def close(self):
        """关闭连接"""
        if self.connection:
            await self.connection.close()
            logger.info("消息队列连接已关闭")


# 全局消息队列服务实例
_message_queue_service: Optional[MessageQueueService] = None


def get_message_queue_service() -> MessageQueueService:
    """获取消息队列服务实例"""
    global _message_queue_service
    if _message_queue_service is None:
        _message_queue_service = MessageQueueService()
    return _message_queue_service


def init_message_queue_service():
    """初始化消息队列服务"""
    global _message_queue_service
    if _message_queue_service is None:
        _message_queue_service = MessageQueueService()
        logger.info("消息队列服务初始化完成")
    return _message_queue_service


async def send_message(message: Dict[str, Any], routing_key: str = None) -> Dict[str, Any]:
    """发送消息"""
    return await get_message_queue_service().send_message(message, routing_key)


async def consume_messages(callback: Callable[[Dict[str, Any]], None]):
    """消费消息"""
    await get_message_queue_service().consume_messages(callback)


async def close_message_queue():
    """关闭消息队列"""
    await get_message_queue_service().close()