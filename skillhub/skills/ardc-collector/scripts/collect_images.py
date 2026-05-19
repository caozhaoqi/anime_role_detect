#!/usr/bin/env python3
"""ARD Collector - 动漫图片采集器"""

import os
import json
import time
import requests
from typing import List, Dict, Any
from pathlib import Path

class ImageCollector:
    """图片采集器"""
    
    def __init__(self, output_dir: str = "data/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_urls = set()
        
        # 加载已下载记录
        self._load_downloaded()
    
    def _load_downloaded(self):
        """加载已下载的 URL 记录"""
        record_file = self.output_dir / ".downloaded.json"
        if record_file.exists():
            with open(record_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.downloaded_urls = set(data.get("urls", []))
    
    def _save_downloaded(self):
        """保存已下载记录"""
        record_file = self.output_dir / ".downloaded.json"
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump({"urls": list(self.downloaded_urls), "updated_at": time.time()}, f)
    
    def download_image(self, url: str, filename: str = None) -> bool:
        """
        下载单张图片
        
        Args:
            url: 图片 URL
            filename: 保存文件名（可选）
        
        Returns:
            是否下载成功
        """
        if url in self.downloaded_urls:
            return False
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            if not filename:
                filename = os.path.basename(url)
                if not filename:
                    filename = f"image_{int(time.time())}.jpg"
            
            filepath = self.output_dir / filename
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            self.downloaded_urls.add(url)
            self._save_downloaded()
            
            return True
        except Exception as e:
            print(f"下载失败 {url}: {e}")
            return False
    
    def download_batch(self, urls: List[str], delay: float = 0.5) -> int:
        """
        批量下载图片
        
        Args:
            urls: 图片 URL 列表
            delay: 下载间隔（秒）
        
        Returns:
            成功下载数量
        """
        success_count = 0
        
        for i, url in enumerate(urls):
            if self.download_image(url):
                success_count += 1
            
            if i < len(urls) - 1 and delay > 0:
                time.sleep(delay)
        
        return success_count
    
    def collect_from_api(self, api_url: str, params: Dict[str, Any] = None) -> int:
        """
        从 API 采集图片
        
        Args:
            api_url: 数据源 API
            params: 请求参数
        
        Returns:
            成功下载数量
        """
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            urls = data.get("images", [])
            return self.download_batch(urls)
        except Exception as e:
            print(f"从 API 采集失败: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取采集统计"""
        total_files = len(list(self.output_dir.glob("*.jpg"))) + len(list(self.output_dir.glob("*.png")))
        return {
            "downloaded_count": len(self.downloaded_urls),
            "saved_files": total_files,
            "output_dir": str(self.output_dir)
        }

if __name__ == "__main__":
    collector = ImageCollector()
    
    # 示例：批量下载
    test_urls = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        "https://example.com/image3.jpg"
    ]
    
    print("开始采集图片...")
    count = collector.download_batch(test_urls)
    print(f"成功下载 {count} 张图片")
    
    stats = collector.get_stats()
    print(f"统计信息: {stats}")
