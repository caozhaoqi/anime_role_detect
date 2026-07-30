#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色数据采集API
集成禁漫屋(JMComic)爬虫系统
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent

import threading
from typing import Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

# 采集系统路径
spider_path = project_root / "archived" / "spider_image_system" / "src"
sys.path.insert(0, str(spider_path))

JM_AVAILABLE = False
JmOption = None
JmDownloader = None
JmApiClient = None

try:
    from jmcomic import JmOption, JmDownloader, JmApiClient
    JM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ JMComic模块导入失败: {e}")

try:
    from ui_event.get_url import spider_artworks_url
    from run import constants as spider_constants
    SPIDER_AVAILABLE = True
except ImportError as e:
    SPIDER_AVAILABLE = False
    print(f"⚠️ Spider系统模块导入失败: {e}")


router = APIRouter(prefix="", tags=["数据采集"])


class AlbumDownloadRequest(BaseModel):
    """本子下载请求"""
    album_id: str
    author: Optional[str] = None
    tags: Optional[list] = []


class SpiderRequest(BaseModel):
    """爬虫请求"""
    keyword: str
    max_count: int = 100


# 全局状态
collector_state = {
    "is_downloading": False,
    "is_spidering": False,
    "current_task": None,
    "downloaded_count": 0,
    "spidered_count": 0
}


def get_jm_option(base_dir: str = "data/raw_dataset") -> JmOption:
    """获取JM配置"""
    if not JM_AVAILABLE:
        raise HTTPException(status_code=500, detail="JMComic模块不可用")
    
    return JmOption(
        photo_order='true',
        create_save_folder='true',
        debug_mode='false',
        base_dir=base_dir,
    )


@router.get("/jm/album/{album_id}", summary="获取本子详情")
async def get_album_detail(album_id: str) -> Dict[str, Any]:
    """获取本子详情"""
    if not JM_AVAILABLE:
        raise HTTPException(status_code=500, detail="JMComic模块不可用")
    
    try:
        client = JmApiClient()
        album = client.get_album_detail(album_id)
        
        return {
            "id": album.id,
            "title": album.title,
            "author": album.author,
            "tags": getattr(album, 'tags', []),
            "page_count": album.page_count,
            "photos": len(album)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取本子详情失败: {str(e)}")


@router.post("/jm/download", summary="下载本子")
async def download_album(request: AlbumDownloadRequest) -> Dict[str, Any]:
    """下载本子"""
    if not JM_AVAILABLE:
        raise HTTPException(status_code=500, detail="JMComic模块不可用")
    
    if collector_state["is_downloading"]:
        raise HTTPException(status_code=409, detail="已有下载任务在进行中")
    
    collector_state["is_downloading"] = True
    collector_state["current_task"] = f"download:{request.album_id}"
    
    try:
        option = get_jm_option()
        downloader = JmDownloader(option)
        
        # 后台执行下载
        def do_download():
            try:
                album = downloader.download_album(request.album_id)
                collector_state["downloaded_count"] += album.page_count
            except Exception as e:
                print(f"下载失败: {e}")
            finally:
                collector_state["is_downloading"] = False
                collector_state["current_task"] = None
        
        thread = threading.Thread(target=do_download)
        thread.start()
        
        return {
            "status": "started",
            "message": f"开始下载本子 {request.album_id}",
            "task_id": f"download:{request.album_id}"
        }
    except Exception as e:
        collector_state["is_downloading"] = False
        raise HTTPException(status_code=500, detail=f"启动下载失败: {str(e)}")


@router.get("/jm/status", summary="获取下载状态")
async def get_download_status() -> Dict[str, Any]:
    """获取当前下载状态"""
    return {
        "is_downloading": collector_state["is_downloading"],
        "is_spidering": collector_state["is_spidering"],
        "current_task": collector_state["current_task"],
        "downloaded_count": collector_state["downloaded_count"],
        "spidered_count": collector_state["spidered_count"]
    }


@router.post("/spider/keyword", summary="爬取关键字图片")
async def spider_by_keyword(request: SpiderRequest) -> Dict[str, Any]:
    """按关键字爬取图片"""
    if collector_state["is_spidering"]:
        raise HTTPException(status_code=409, detail="已有爬虫任务在进行中")
    
    if not request.keyword:
        raise HTTPException(status_code=400, detail="关键字不能为空")
    
    collector_state["is_spidering"] = True
    collector_state["current_task"] = f"spider:{request.keyword}"
    
    try:
        spider_constants.SpiderConfig.spider_mode = 'manual'
        spider_constants.SpiderConfig.max_urls_per_keyword = request.max_count
        
        def do_spider():
            try:
                spider_artworks_url(None, request.keyword)
            except Exception as e:
                print(f"爬取失败: {e}")
            finally:
                collector_state["is_spidering"] = False
                collector_state["current_task"] = None
        
        thread = threading.Thread(target=do_spider)
        thread.start()
        
        return {
            "status": "started",
            "message": f"开始爬取关键字: {request.keyword}",
            "task_id": f"spider:{request.keyword}",
            "max_count": request.max_count
        }
    except Exception as e:
        collector_state["is_spidering"] = False
        raise HTTPException(status_code=500, detail=f"启动爬虫失败: {str(e)}")


@router.post("/spider/stop", summary="停止爬虫")
async def stop_spider() -> Dict[str, str]:
    """停止爬虫任务"""
    spider_constants.SpiderConfig.stop_spider_url_flag = True
    collector_state["is_spidering"] = False
    collector_state["current_task"] = None
    
    return {"status": "stopped", "message": "爬虫已停止"}


@router.post("/download/stop", summary="停止下载")
async def stop_download() -> Dict[str, str]:
    """停止下载任务"""
    spider_constants.SpiderConfig.stop_download_image_flag = True
    collector_state["is_downloading"] = False
    collector_state["current_task"] = None
    
    return {"status": "stopped", "message": "下载已停止"}


@router.get("/jm/domain/test", summary="测试JM可用域名")
async def test_jm_domain() -> Dict[str, Any]:
    """测试JM可用域名"""
    if not JM_AVAILABLE:
        raise HTTPException(status_code=500, detail="JMComic模块不可用")
    
    try:
        from utils.jm_domain_detect import jm_domain_test
        
        result = {"domains": [], "working": None}
        
        def do_test():
            try:
                result["domains"] = jm_domain_test()
                result["working"] = result["domains"][0] if result["domains"] else None
            except Exception as e:
                print(f"域名检测失败: {e}")
        
        thread = threading.Thread(target=do_test)
        thread.start()
        thread.join(timeout=30)
        
        return {
            "status": "completed",
            "domains": result["domains"],
            "working_domain": result["working"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"域名检测失败: {str(e)}")
