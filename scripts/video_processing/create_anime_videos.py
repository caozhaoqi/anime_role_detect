#!/usr/bin/env python3
"""
创建真实二次元动漫测试视频脚本
使用真实的二次元角色图像创建测试视频
"""
import os
import cv2
import numpy as np
import argparse
import logging
import random
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('create_anime_videos')


class AnimeVideoCreator:
    """二次元动漫视频创建器"""
    
    def __init__(self, output_dir='data/videos', image_dir='data/all_characters'):
        """初始化创建器
        
        Args:
            output_dir: 输出目录
            image_dir: 图像目录
        """
        self.output_dir = output_dir
        self.image_dir = image_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"视频存储目录: {self.output_dir}")
        logger.info(f"图像来源目录: {self.image_dir}")
    
    def get_character_images(self):
        """获取角色图像
        
        Returns:
            角色图像字典 {角色名称: [图像路径列表]}
        """
        character_images = {}
        
        try:
            # 遍历角色目录
            for character in os.listdir(self.image_dir):
                character_path = os.path.join(self.image_dir, character)
                if os.path.isdir(character_path):
                    # 获取该角色的所有图像
                    images = []
                    for img_name in os.listdir(character_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            images.append(os.path.join(character_path, img_name))
                    
                    if images:
                        character_images[character] = images
                        logger.info(f"找到角色 {character} 的 {len(images)} 张图像")
        
        except Exception as e:
            logger.error(f"获取角色图像时出错: {e}")
        
        return character_images
    
    def create_anime_video(self, filename, duration=10, fps=30):
        """创建二次元动漫测试视频
        
        Args:
            filename: 视频文件名
            duration: 视频时长（秒）
            fps: 帧率
            
        Returns:
            创建的视频路径
        """
        try:
            # 获取角色图像
            character_images = self.get_character_images()
            
            if not character_images:
                logger.error('未找到角色图像，使用默认模式创建视频')
                return self.create_default_anime_video(filename, duration, fps)
            
            # 视频参数
            width, height = 1280, 720
            total_frames = duration * fps
            
            # 创建视频编写器
            output_path = os.path.join(self.output_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 获取所有角色和图像
            characters = list(character_images.keys())
            all_images = []
            for char, imgs in character_images.items():
                for img in imgs[:10]:  # 每个角色最多使用10张图像
                    all_images.append((char, img))
            
            if not all_images:
                logger.error('未找到有效图像，使用默认模式创建视频')
                return self.create_default_anime_video(filename, duration, fps)
            
            logger.info(f"开始创建二次元动漫测试视频: {filename}")
            logger.info(f"视频参数: {width}x{height}, {fps}fps, {duration}秒")
            logger.info(f"使用 {len(characters)} 个角色，共 {len(all_images)} 张图像")
            
            for frame_idx in range(total_frames):
                # 随机选择角色和图像
                char, img_path = random.choice(all_images)
                
                try:
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    # 调整图像大小
                    img = img.resize((width, height), Image.LANCZOS)
                    
                    # 转换为OpenCV格式
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    
                    # 在帧上添加角色信息
                    cv2.putText(frame, f"Character: {char}", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.5, (255, 255, 255), 3)
                    
                    cv2.putText(frame, f"Frame: {frame_idx+1}/{total_frames}", 
                               (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2)
                    
                    cv2.putText(frame, "Anime Character Test Video", 
                               (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.2, (255, 255, 255), 2)
                    
                except Exception as e:
                    logger.warning(f"处理图像 {img_path} 时出错: {e}")
                    # 创建默认帧
                    bg_color = np.random.randint(0, 255, 3).tolist()
                    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
                    cv2.putText(frame, f"Character: {char}", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.5, (255, 255, 255), 3)
                
                # 写入帧
                out.write(frame)
            
            # 释放资源
            out.release()
            
            logger.info(f"二次元动漫测试视频创建成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"创建视频时出错: {e}")
            return None
    
    def create_default_anime_video(self, filename, duration=10, fps=30):
        """创建默认二次元动漫测试视频
        
        Args:
            filename: 视频文件名
            duration: 视频时长（秒）
            fps: 帧率
            
        Returns:
            创建的视频路径
        """
        try:
            # 视频参数
            width, height = 1280, 720
            total_frames = duration * fps
            
            # 创建视频编写器
            output_path = os.path.join(self.output_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 二次元角色列表
            anime_characters = [
                "原神_温迪", "原神_胡桃", "原神_可莉", "原神_魈",
                "崩坏三_琪亚娜", "崩坏三_芽衣", "崩坏三_布洛妮娅",
                "初音未来", "约会大作战_时崎狂三", "约会大作战_五河琴里",
                "Re0_雷姆", "Re0_拉姆", "冰菓_千反田爱瑠",
                "东京复仇者_佐野万次郎", "东京复仇者_龙宫寺坚"
            ]
            
            logger.info(f"使用默认模式创建二次元动漫测试视频: {filename}")
            logger.info(f"视频参数: {width}x{height}, {fps}fps, {duration}秒")
            logger.info(f"包含 {len(anime_characters)} 个二次元角色")
            
            for frame_idx in range(total_frames):
                # 创建渐变背景
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(width):
                    for j in range(height):
                        r = int(100 + 155 * (i/width))
                        g = int(50 + 205 * (j/height))
                        b = int(150 + 105 * ((i+j)/(width+height)))
                        frame[j, i] = [b, g, r]  # BGR格式
                
                # 随机选择角色
                character = anime_characters[frame_idx % len(anime_characters)]
                
                # 绘制角色信息
                cv2.putText(frame, f"Character: {character}", 
                           (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                           2, (255, 255, 255), 4)
                
                cv2.putText(frame, f"Frame: {frame_idx+1}/{total_frames}", 
                           (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.2, (255, 255, 255), 2)
                
                cv2.putText(frame, "Anime Character Test Video", 
                           (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.5, (255, 255, 255), 3)
                
                cv2.putText(frame, "Real Anime Style", 
                           (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.2, (255, 255, 255), 2)
                
                # 绘制动漫风格的装饰元素
                for i in range(5):
                    x = int(width * 0.7 + 100 * np.sin(frame_idx * 0.1 + i))
                    y = int(height * 0.5 + 100 * np.cos(frame_idx * 0.15 + i))
                    size = 20 + i * 10
                    color = (255, 255 - i * 40, 255 - i * 40)
                    cv2.circle(frame, (x, y), size, color, -1)
                
                # 写入帧
                out.write(frame)
            
            # 释放资源
            out.release()
            
            logger.info(f"默认二次元动漫测试视频创建成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"创建默认视频时出错: {e}")
            return None
    
    def create_multiple_videos(self, count=3):
        """创建多个二次元动漫测试视频
        
        Args:
            count: 视频数量
            
        Returns:
            创建的视频路径列表
        """
        created_videos = []
        
        for i in range(count):
            filename = f"anime_character_real_{i+1}.mp4"
            video_path = self.create_anime_video(filename)
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
    parser = argparse.ArgumentParser(description='二次元动漫视频创建工具')
    
    parser.add_argument('--output_dir', type=str,
                        default='data/videos',
                        help='输出目录')
    parser.add_argument('--image_dir', type=str,
                        default='data/all_characters',
                        help='角色图像目录')
    parser.add_argument('--count', type=int,
                        default=3,
                        help='视频数量')
    parser.add_argument('--test', action='store_true',
                        help='测试创建的视频')
    
    args = parser.parse_args()
    
    logger.info('开始创建二次元动漫测试视频...')
    
    # 创建视频创建器
    creator = AnimeVideoCreator(args.output_dir, args.image_dir)
    
    # 创建测试视频
    created_videos = creator.create_multiple_videos(args.count)
    
    if created_videos:
        logger.info(f"成功创建 {len(created_videos)} 个二次元动漫测试视频")
        
        # 测试视频
        if args.test:
            logger.info('开始测试视频...')
            creator.test_videos(created_videos)
    else:
        logger.error('未成功创建任何视频')
    
    logger.info('二次元动漫视频创建完成！')


if __name__ == "__main__":
    main()