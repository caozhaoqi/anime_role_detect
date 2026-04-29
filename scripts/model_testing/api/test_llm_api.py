#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试大模型API连接
"""

import os
import requests
import time

def test_api_connection():
    """测试API连接"""
    # 从.env文件读取配置
    env_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/.env"
    api_key = ""
    api_base = "https://api.siliconflow.cn/v1"
    model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    if key == 'OPENAI_API_KEY':
                        api_key = value.strip()
                    elif key == 'OPENAI_API_BASE':
                        api_base = value.strip()
                    elif key == 'MODEL_NAME':
                        model_name = value.strip()
    
    print(f"API配置:")
    print(f"  API Key: {api_key[:10]}..." if len(api_key) > 10 else f"  API Key: {api_key}")
    print(f"  API Base: {api_base}")
    print(f"  Model: {model_name}")
    
    if not api_key:
        print("错误: 未找到API Key")
        return False
    
    # 测试不同的提示词
    test_prompts = [
        ("简单测试", "你好"),
        ("角色列表-原神", "列出原神游戏中的5个主要角色，每行一个，不要编号"),
        ("角色列表-星穹铁道", "列出星穹铁道游戏中的5个主要角色，每行一个，不要编号"),
        ("角色列表-崩坏3", "列出崩坏3游戏中的5个主要角色，每行一个，不要编号"),
    ]
    
    api_url = f"{api_base}/chat/completions"
    
    for test_name, prompt in test_prompts:
        print(f"\n测试: {test_name}")
        print(f"提示词: {prompt}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "你是一个游戏角色专家"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        try:
            start_time = time.time()
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            elapsed_time = time.time() - start_time
            
            print(f"响应时间: {elapsed_time:.2f}秒")
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                print(f"响应内容: {content[:100]}...")
                print("✅ 成功")
            else:
                print(f"❌ 失败: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print("❌ 超时")
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        # 避免请求过于频繁
        time.sleep(2)
    
    return True

if __name__ == "__main__":
    test_api_connection()