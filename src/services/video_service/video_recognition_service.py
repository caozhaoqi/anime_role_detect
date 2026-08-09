#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频实时抽帧识别库（进程内使用，非独立部署服务）

支持实时视频流抽帧和角色识别，用于弹幕实时显示角色信息。

注意：本模块**不**作为独立 HTTP 服务部署。历史上曾存在
``video_service_app.py`` 将其包装为 FastAPI 服务，但该入口从未被 supervisord/
compose/K8s 启动，且视频路由已由 multimedia-service 的 ``/video/*`` 端点接管，
故于 2026-08-09 删除。当前本模块仅由 api-service 的 ``video_routes`` /
``search_routes`` 通过延迟 import 以**库**的形式调用。
"""

import os
import sys
import cv2
import time
import threading
import queue
from typing import List, Dict, Optional, Tuple
from PIL import Image

# 添加项目根目录到Python路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# 延迟导入分类服务，避免启动时的锁竞争
classify_image = None

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("video_recognition_service")


def _import_classify():
    """延迟导入分类服务"""
    global classify_image
    if classify_image is None:
        logger.info("延迟导入分类服务...")
        from src.services.model.classification_service import classify_image

        logger.info("分类服务导入完成")


class VideoRecognitionService:
    """
    视频实时抽帧识别服务

    功能：
    1. 从视频流中按时间间隔抽帧
    2. 对每一帧进行角色识别
    3. 输出识别结果（可用于弹幕显示）
    4. 支持实时处理和批量处理模式
    """

    def __init__(
        self,
        frame_interval: float = 1.0,  # 抽帧间隔（秒）
        max_queue_size: int = 30,
        confidence_threshold: float = 0.5,
        min_detection_interval: float = 2.0,
    ):
        """
        初始化视频识别服务

        Args:
            frame_interval: 抽帧间隔（秒）
            max_queue_size: 处理队列最大大小
            confidence_threshold: 识别置信度阈值
            min_detection_interval: 同一角色最小识别间隔（秒）
        """
        self.frame_interval = frame_interval
        self.max_queue_size = max_queue_size
        self.confidence_threshold = confidence_threshold
        self.min_detection_interval = min_detection_interval

        # 队列和线程
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_thread = None
        self.running = False

        # 状态跟踪
        self.last_detection = {}  # {角色名: 最后检测时间}
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = None

        # 回调函数
        self.on_result_callback = None

    def set_result_callback(self, callback):
        """设置结果回调函数"""
        self.on_result_callback = callback

    def _process_frame(self, frame: cv2.Mat, timestamp: float) -> Optional[Dict]:
        """
        处理单帧图像

        Args:
            frame: 帧图像
            timestamp: 时间戳（秒）

        Returns:
            识别结果或None
        """
        try:
            # 延迟导入分类服务
            _import_classify()

            # 将OpenCV图像转换为PIL图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # 保存临时文件用于分类
            temp_path = f"/tmp/video_frame_{int(time.time() * 1000)}.jpg"
            pil_image.save(temp_path)

            # 使用分类服务识别角色
            role, similarity, boxes, mode, attributes, text_detections = classify_image(
                temp_path, use_model=True, use_attributes=True
            )

            # 清理临时文件
            os.remove(temp_path)

            # 过滤低置信度结果
            if similarity < self.confidence_threshold:
                return None

            # 检查是否需要去重（同一角色短时间内重复检测）
            if role in self.last_detection:
                time_since_last = timestamp - self.last_detection[role]
                if time_since_last < self.min_detection_interval:
                    logger.debug(f"跳过重复检测: {role} (上次检测: {time_since_last:.2f}s前)")
                    return None

            # 更新最后检测时间
            self.last_detection[role] = timestamp

            return {
                "timestamp": timestamp,
                "role": role,
                "similarity": similarity,
                "boxes": boxes,
                "mode": mode,
                "attributes": attributes,
                "frame_number": self.frame_count,
            }
        except Exception as e:
            logger.error(f"处理帧失败: {e}")
            return None

    def _processing_worker(self):
        """处理线程工作函数"""
        logger.info("处理线程启动")

        while self.running or not self.frame_queue.empty():
            try:
                frame, timestamp = self.frame_queue.get(timeout=1.0)

                # 处理帧
                result = self._process_frame(frame, timestamp)

                if result:
                    # 发送到结果队列
                    if not self.result_queue.full():
                        self.result_queue.put(result)

                    # 调用回调函数
                    if self.on_result_callback:
                        try:
                            self.on_result_callback(result)
                        except Exception as e:
                            logger.error(f"回调函数执行失败: {e}")

                self.processed_count += 1
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理队列失败: {e}")

        logger.info("处理线程停止")

    def start(self):
        """启动服务"""
        if self.running:
            logger.warning("服务已运行")
            return

        self.running = True
        self.start_time = time.time()
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        logger.info("视频识别服务启动")

    def stop(self):
        """停止服务"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("视频识别服务停止")

    def process_video_file(self, video_path: str, output_callback=None) -> List[Dict]:
        """
        处理视频文件（批量模式）

        Args:
            video_path: 视频文件路径
            output_callback: 每帧结果回调

        Returns:
            所有识别结果列表
        """
        logger.info(f"开始处理视频文件: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_frames = int(fps * self.frame_interval)

        results = []
        self.frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 按间隔抽帧
                if self.frame_count % frame_interval_frames == 0:
                    timestamp = self.frame_count / fps
                    result = self._process_frame(frame, timestamp)

                    if result:
                        results.append(result)

                        if output_callback:
                            output_callback(result)

                        logger.info(
                            f"[{timestamp:.2f}s] 检测到角色: {result['role']} (相似度: {result['similarity']:.4f})"
                        )

                self.frame_count += 1

                # 进度输出
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100
                    logger.info(f"处理进度: {progress:.1f}%")

        finally:
            cap.release()

        logger.info(f"视频处理完成，共检测到 {len(results)} 个角色")
        return results

    def process_realtime(self, video_source=0):
        """
        处理实时视频流（摄像头或视频流）

        Args:
            video_source: 视频源（0为默认摄像头，或RTSP URL）
        """
        logger.info(f"开始处理实时视频流: {video_source}")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval_frames = int(fps * self.frame_interval)

        self.start()

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("无法读取帧，重试...")
                    time.sleep(0.1)
                    continue

                # 按间隔抽帧
                if self.frame_count % frame_interval_frames == 0:
                    timestamp = self.frame_count / fps

                    # 添加到队列
                    if not self.frame_queue.full():
                        self.frame_queue.put((frame.copy(), timestamp))
                    else:
                        logger.warning("帧队列已满，丢弃帧")

                self.frame_count += 1

                # 显示预览（可选）
                # cv2.imshow('Video Recognition', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        finally:
            self.stop()
            cap.release()
            # cv2.destroyAllWindows()

    def get_stats(self) -> Dict:
        """获取服务统计信息"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "running": self.running,
            "frame_count": self.frame_count,
            "processed_count": self.processed_count,
            "queue_size": self.frame_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "elapsed_time": elapsed,
            "fps": self.processed_count / elapsed if elapsed > 0 else 0,
        }


def demo_callback(result: Dict):
    """示例回调函数 - 模拟弹幕显示"""
    print(f"\n[弹幕] [{result['timestamp']:.2f}s] 识别到角色: {result['role']}")
    if result["attributes"]:
        tags = [attr["tag"] for attr in result["attributes"][:3]]
        print(f"      属性: {', '.join(tags)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频实时抽帧识别服务")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--realtime", action="store_true", help="实时模式（摄像头）")
    parser.add_argument("--interval", type=float, default=1.0, help="抽帧间隔（秒）")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")

    args = parser.parse_args()

    # 创建服务
    service = VideoRecognitionService(
        frame_interval=args.interval, confidence_threshold=args.threshold
    )

    # 设置回调函数
    service.set_result_callback(demo_callback)

    try:
        if args.video:
            # 处理视频文件
            results = service.process_video_file(args.video)

            # 输出汇总
            print("\n" + "=" * 50)
            print("视频识别完成")
            print("=" * 50)
            print(f"处理帧数: {service.frame_count}")
            print(f"识别结果: {len(results)}")
            print("\n识别到的角色:")
            roles_found = {}
            for r in results:
                roles_found[r["role"]] = roles_found.get(r["role"], 0) + 1

            for role, count in sorted(roles_found.items(), key=lambda x: x[1], reverse=True):
                print(f"  {role}: {count} 次")

        elif args.realtime:
            # 实时模式
            print("实时视频识别开始，按Ctrl+C退出...")
            service.process_realtime()

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        service.stop()
