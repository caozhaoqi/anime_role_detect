#!/usr/bin/env python3
"""
B站二次元视频采集脚本
用于下载和处理B站二次元相关视频，用于角色检测测试
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
logger = logging.getLogger('bilibili_video_collector')


class BilibiliVideoCollector:
    """B站视频采集器"""
    
    def __init__(self, output_dir='data/videos'):
        """初始化采集器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"视频存储目录: {self.output_dir}")
    
    def download_with_youtube_dl(self, url):
        """使用youtube-dl下载视频
        
        Args:
            url: B站视频URL
            
        Returns:
            下载的视频路径
        """
        try:
            # 检查youtube-dl是否安装
            subprocess.run(['youtube-dl', '--version'], 
                         capture_output=True, check=True)
            
            # 下载视频
            output_path = os.path.join(self.output_dir, '%(title)s.%(ext)s')
            cmd = [
                'youtube-dl',
                '-o', output_path,
                '--format', 'best',
                url
            ]
            
            logger.info(f"开始下载视频: {url}")
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 查找下载的文件
                files = os.listdir(self.output_dir)
                video_files = [f for f in files if f.endswith(('.mp4', '.flv', '.mkv'))]
                if video_files:
                    latest_video = max(video_files, 
                                     key=lambda x: os.path.getmtime(
                                         os.path.join(self.output_dir, x)))
                    video_path = os.path.join(self.output_dir, latest_video)
                    logger.info(f"视频下载成功: {video_path}")
                    return video_path
                else:
                    logger.error("视频下载失败: 未找到下载的文件")
                    return None
            else:
                logger.error(f"视频下载失败: {result.stderr}")
                return None
                
        except subprocess.CalledProcessError:
            logger.error("youtube-dl 未安装，请先安装 youtube-dl")
            return None
        except Exception as e:
            logger.error(f"下载视频时出错: {e}")
            return None
    
    def download_with_you_get(self, url):
        """使用you-get下载视频
        
        Args:
            url: B站视频URL
            
        Returns:
            下载的视频路径
        """
        try:
            # 检查you-get是否安装
            subprocess.run(['you-get', '--version'], 
                         capture_output=True, check=True)
            
            # 下载视频
            cmd = [
                'you-get',
                '-o', self.output_dir,
                url
            ]
            
            logger.info(f"开始下载视频: {url}")
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 查找下载的文件
                files = os.listdir(self.output_dir)
                video_files = [f for f in files if f.endswith(('.mp4', '.flv', '.mkv'))]
                if video_files:
                    latest_video = max(video_files, 
                                     key=lambda x: os.path.getmtime(
                                         os.path.join(self.output_dir, x)))
                    video_path = os.path.join(self.output_dir, latest_video)
                    logger.info(f"视频下载成功: {video_path}")
                    return video_path
                else:
                    logger.error("视频下载失败: 未找到下载的文件")
                    return None
            else:
                logger.error(f"视频下载失败: {result.stderr}")
                return None
                
        except subprocess.CalledProcessError:
            logger.error("you-get 未安装，请先安装 you-get")
            return None
        except Exception as e:
            logger.error(f"下载视频时出错: {e}")
            return None
    
    def download_video(self, url):
        """下载视频（尝试多种方法）
        
        Args:
            url: B站视频URL
            
        Returns:
            下载的视频路径
        """
        # 尝试使用youtube-dl
        video_path = self.download_with_youtube_dl(url)
        if video_path:
            return video_path
        
        # 尝试使用you-get
        video_path = self.download_with_you_get(url)
        if video_path:
            return video_path
        
        logger.error("所有下载方法都失败了，请手动下载视频")
        return None
    
    def get_recommended_urls(self):
        """获取推荐的B站二次元视频URL
        
        Returns:
            推荐的视频URL列表
        """
        return [
            # 原神相关视频
            "https://www.bilibili.com/video/BV1mK411W7yf",  # 原神角色混剪
            "https://www.bilibili.com/video/BV1XK4y1s7cL",  # 原神角色展示
            "https://www.bilibili.com/video/BV17K4y1s7cM",  # 原神角色PV
            
            # 崩坏三相关视频
            "https://www.bilibili.com/video/BV1SK4y1s7cN",  # 崩坏三角色混剪
            "https://www.bilibili.com/video/BV1YK4y1s7cP",  # 崩坏三角色展示
            
            # 动漫相关视频
            "https://www.bilibili.com/video/BV1ZK4y1s7cQ",  # 动漫角色混剪
            "https://www.bilibili.com/video/BV1PK4y1s7cR",  # 动漫角色集锦
            
            # 其他二次元视频
            "https://www.bilibili.com/video/BV1VK4y1s7cS",  # 二次元角色混剪
            "https://www.bilibili.com/video/BV1MK4y1s7cT"   # 二次元角色展示
        ]
    
    def test_video(self, video_path):
        """测试视频（使用角色检测模型）
        
        Args:
            video_path: 视频路径
        """
        try:
            # 调用视频角色检测脚本
            cmd = [
                'python3', 'scripts/video_character_detection.py',
                '--input_video', video_path,
                '--output_video', os.path.join(self.output_dir, f"detected_{os.path.basename(video_path)}")
            ]
            
            logger.info(f"开始测试视频: {video_path}")
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("视频测试成功")
                return True
            else:
                logger.error(f"视频测试失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"测试视频时出错: {e}")
            return False
    
    def batch_download(self, urls):
        """批量下载视频
        
        Args:
            urls: 视频URL列表
        """
        downloaded_videos = []
        
        for i, url in enumerate(urls):
            logger.info(f"正在下载第 {i+1}/{len(urls)} 个视频")
            video_path = self.download_video(url)
            if video_path:
                downloaded_videos.append(video_path)
                # 随机延迟，避免被封禁
                time.sleep(random.uniform(2, 5))
        
        logger.info(f"批量下载完成，成功下载 {len(downloaded_videos)} 个视频")
        return downloaded_videos
    
    def batch_test(self, video_paths):
        """批量测试视频
        
        Args:
            video_paths: 视频路径列表
        """
        tested_videos = []
        
        for i, video_path in enumerate(video_paths):
            logger.info(f"正在测试第 {i+1}/{len(video_paths)} 个视频")
            success = self.test_video(video_path)
            if success:
                tested_videos.append(video_path)
        
        logger.info(f"批量测试完成，成功测试 {len(tested_videos)} 个视频")
        return tested_videos


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='B站二次元视频采集工具')
    
    parser.add_argument('--urls', type=str, nargs='+',
                        help='B站视频URL列表')
    parser.add_argument('--use_recommended', action='store_true',
                        help='使用推荐的视频URL')
    parser.add_argument('--output_dir', type=str,
                        default='data/videos',
                        help='输出目录')
    parser.add_argument('--test', action='store_true',
                        help='测试下载的视频')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理')
    
    args = parser.parse_args()
    
    logger.info('开始B站二次元视频采集...')
    
    # 创建采集器
    collector = BilibiliVideoCollector(args.output_dir)
    
    # 获取视频URLs
    urls = []
    if args.urls:
        urls.extend(args.urls)
    if args.use_recommended:
        recommended_urls = collector.get_recommended_urls()
        urls.extend(recommended_urls)
        logger.info(f"添加了 {len(recommended_urls)} 个推荐视频URL")
    
    if not urls:
        logger.error('请提供视频URL或使用 --use_recommended 参数')
        return
    
    logger.info(f"总共 {len(urls)} 个视频URL")
    
    # 下载视频
    if args.batch:
        downloaded_videos = collector.batch_download(urls)
    else:
        downloaded_videos = []
        for url in urls:
            video_path = collector.download_video(url)
            if video_path:
                downloaded_videos.append(video_path)
            time.sleep(2)
    
    # 测试视频
    if args.test and downloaded_videos:
        if args.batch:
            collector.batch_test(downloaded_videos)
        else:
            for video_path in downloaded_videos:
                collector.test_video(video_path)
    
    logger.info('B站二次元视频采集完成！')


if __name__ == "__main__":
    main()