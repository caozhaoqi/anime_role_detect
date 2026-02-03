#!/usr/bin/env python3
"""
创建测试视频脚本
用于生成测试用的二次元角色视频
"""
import os
import cv2
import numpy as np
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('create_test_videos')


class TestVideoCreator:
    """测试视频创建器"""
    
    def __init__(self, output_dir='data/videos'):
        """初始化创建器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"视频存储目录: {self.output_dir}")
    
    def create_anime_character_video(self, filename, duration=5, fps=30):
        """创建二次元角色测试视频
        
        Args:
            filename: 视频文件名
            duration: 视频时长（秒）
            fps: 帧率
            
        Returns:
            创建的视频路径
        """
        try:
            # 视频参数
            width, height = 640, 480
            total_frames = duration * fps
            
            # 创建视频编写器
            output_path = os.path.join(self.output_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 角色列表
            characters = [
                "原神_温迪", "原神_胡桃", "原神_可莉",
                "崩坏三_琪亚娜", "初音未来", "约会大作战_时崎狂三"
            ]
            
            logger.info(f"开始创建测试视频: {filename}")
            logger.info(f"视频参数: {width}x{height}, {fps}fps, {duration}秒")
            
            for frame_idx in range(total_frames):
                # 创建随机背景
                bg_color = np.random.randint(0, 255, 3).tolist()
                frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
                
                # 随机选择角色
                character = characters[frame_idx % len(characters)]
                
                # 在帧上绘制角色信息
                cv2.putText(frame, f"Character: {character}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Frame: {frame_idx+1}/{total_frames}", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, "Anime Character Test Video", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                # 添加一些随机图形模拟角色
                for i in range(5):
                    x = np.random.randint(100, width-100)
                    y = np.random.randint(250, height-100)
                    size = np.random.randint(20, 50)
                    color = np.random.randint(0, 255, 3).tolist()
                    cv2.circle(frame, (x, y), size, color, -1)
                
                # 写入帧
                out.write(frame)
            
            # 释放资源
            out.release()
            
            logger.info(f"测试视频创建成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"创建视频时出错: {e}")
            return None
    
    def create_multiple_videos(self, count=3):
        """创建多个测试视频
        
        Args:
            count: 视频数量
            
        Returns:
            创建的视频路径列表
        """
        created_videos = []
        
        for i in range(count):
            filename = f"anime_character_test_{i+1}.mp4"
            video_path = self.create_anime_character_video(filename)
            if video_path:
                created_videos.append(video_path)
        
        return created_videos
    
    def test_videos(self, video_paths):
        """测试视频
        
        Args:
            video_paths: 视频路径列表
        """
        for i, video_path in enumerate(video_paths):
            logger.info(f"测试视频 {i+1}/{len(video_paths)}: {video_path}")
            
            # 调用视频角色检测脚本
            output_video = os.path.join(self.output_dir, f"detected_{os.path.basename(video_path)}")
            
            cmd = [
                'python3', 'scripts/video_character_detection.py',
                '--input_video', video_path,
                '--output_video', output_video
            ]
            
            try:
                import subprocess
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"视频测试成功: {output_video}")
                else:
                    logger.error(f"视频测试失败: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"测试视频时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建测试视频工具')
    
    parser.add_argument('--output_dir', type=str,
                        default='data/videos',
                        help='输出目录')
    parser.add_argument('--count', type=int,
                        default=3,
                        help='视频数量')
    parser.add_argument('--test', action='store_true',
                        help='测试创建的视频')
    
    args = parser.parse_args()
    
    logger.info('开始创建测试视频...')
    
    # 创建视频创建器
    creator = TestVideoCreator(args.output_dir)
    
    # 创建测试视频
    created_videos = creator.create_multiple_videos(args.count)
    
    if created_videos:
        logger.info(f"成功创建 {len(created_videos)} 个测试视频")
        
        # 测试视频
        if args.test:
            logger.info('开始测试视频...')
            creator.test_videos(created_videos)
    else:
        logger.error('未成功创建任何视频')
    
    logger.info('测试视频创建完成！')


if __name__ == "__main__":
    main()