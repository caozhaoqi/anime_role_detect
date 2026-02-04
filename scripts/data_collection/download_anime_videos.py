#!/usr/bin/env python3
"""
二次元角色视频下载脚本
用于下载二次元相关视频，用于角色检测测试
"""
import os
import subprocess
import argparse
import logging
import time
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('anime_video_downloader')


class AnimeVideoDownloader:
    """二次元视频下载器"""
    
    def __init__(self, output_dir='data/videos'):
        """初始化下载器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"视频存储目录: {self.output_dir}")
    
    def download_sample_videos(self):
        """下载示例二次元视频
        
        Returns:
            下载的视频路径列表
        """
        sample_videos = [
            # 原神角色展示视频
            "https://www.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
            # 崩坏三角色展示视频
            "https://www.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4",
            # 动漫角色展示视频
            "https://www.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_5mb.mp4"
        ]
        
        downloaded_videos = []
        
        for i, url in enumerate(sample_videos):
            video_path = self.download_video(url, f"anime_sample_{i+1}.mp4")
            if video_path:
                downloaded_videos.append(video_path)
                # 随机延迟
                time.sleep(random.uniform(1, 3))
        
        return downloaded_videos
    
    def download_video(self, url, filename):
        """下载视频
        
        Args:
            url: 视频URL
            filename: 保存的文件名
            
        Returns:
            下载的视频路径
        """
        try:
            output_path = os.path.join(self.output_dir, filename)
            
            # 使用curl下载
            cmd = ['curl', '-L', '-o', output_path, url]
            
            logger.info(f"开始下载视频: {url}")
            logger.info(f"保存到: {output_path}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"视频下载成功: {output_path}")
                    return output_path
                else:
                    logger.error("视频下载失败: 文件为空或不存在")
                    return None
            else:
                logger.error(f"视频下载失败: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"下载视频时出错: {e}")
            return None
    
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
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"视频测试成功: {output_video}")
                else:
                    logger.error(f"视频测试失败: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"测试视频时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='二次元视频下载工具')
    
    parser.add_argument('--output_dir', type=str,
                        default='data/videos',
                        help='输出目录')
    parser.add_argument('--test', action='store_true',
                        help='测试下载的视频')
    
    args = parser.parse_args()
    
    logger.info('开始下载二次元角色视频...')
    
    # 创建下载器
    downloader = AnimeVideoDownloader(args.output_dir)
    
    # 下载示例视频
    downloaded_videos = downloader.download_sample_videos()
    
    if downloaded_videos:
        logger.info(f"成功下载 {len(downloaded_videos)} 个视频")
        
        # 测试视频
        if args.test:
            logger.info('开始测试视频...')
            downloader.test_videos(downloaded_videos)
    else:
        logger.error('未成功下载任何视频')
    
    logger.info('二次元视频下载完成！')


if __name__ == "__main__":
    main()