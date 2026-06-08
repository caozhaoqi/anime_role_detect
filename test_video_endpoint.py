#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频识别端点测试脚本
"""

import requests
import os

# API 端点
API_URL = "http://127.0.0.1:8001/api/video/recognize"

def test_video_recognition():
    """测试视频识别端点"""
    print("=" * 50)
    print("视频识别端点测试")
    print("=" * 50)
    
    # 检查是否有测试视频文件
    test_video_path = "test_video.mp4"
    
    if not os.path.exists(test_video_path):
        print(f"❌ 测试视频文件不存在：{test_video_path}")
        print("\n提示：请上传一个测试视频文件到项目根目录")
        return
    
    print(f"\n📹 测试视频：{test_video_path}")
    
    # 准备请求
    files = {"file": open(test_video_path, "rb")}
    data = {
        "frame_interval": 1.0,
        "confidence_threshold": 0.5
    }
    
    try:
        print("\n🚀 发送请求...")
        response = requests.post(API_URL, files=files, data=data)
        
        print(f"\n📊 响应状态码：{response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 请求成功!")
            print(f"\n响应数据:")
            print(f"  - 成功：{result.get('success')}")
            print(f"  - 消息：{result.get('message')}")
            
            if result.get('data'):
                data = result['data']
                print(f"\n详细数据:")
                print(f"  - 总帧数：{data.get('total_frames_processed')}")
                print(f"  - 检测数：{data.get('total_detections')}")
                print(f"  - 找到的角色：{data.get('roles_found')}")
                print(f"  - 结果数：{len(data.get('results', []))}")
                
                if data.get('results'):
                    print(f"\n前 3 个检测结果:")
                    for i, res in enumerate(data['results'][:3]):
                        print(f"  [{i+1}] 帧 {res.get('frame_index')} @ {res.get('timestamp'):.2f}s")
                        for role in res.get('roles', []):
                            print(f"      - {role.get('role')} ({role.get('similarity'):.2%})")
        else:
            print(f"\n❌ 请求失败：{response.text}")
            
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
    finally:
        if 'files' in locals():
            files['file'].close()
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_video_recognition()
