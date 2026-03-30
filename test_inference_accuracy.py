#!/usr/bin/env python3
"""
使用训练图片测试推理准确性
"""

import os
import requests
import json
import time

# 配置
BASE_URL = 'http://localhost:8000/api/classify'
TRAIN_DIR = 'data/train'

# 统计变量
correct = 0
total = 0

# 遍历训练目录
print("开始测试推理准确性...")
print("=" * 60)

for role in os.listdir(TRAIN_DIR):
    role_dir = os.path.join(TRAIN_DIR, role)
    if os.path.isdir(role_dir):
        print(f"\n测试角色: {role}")
        print("-" * 40)
        
        for img in os.listdir(role_dir):
            if img.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(role_dir, img)
                print(f"测试: {role}/{img}")
                
                # 发送请求
                try:
                    files = {'file': open(img_path, 'rb')}
                    response = requests.post(BASE_URL, files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        predicted_role = result.get('role', 'unknown')
                        similarity = result.get('similarity', 0.0)
                        
                        print(f"  预测: {predicted_role}, 相似度: {similarity:.4f}")
                        
                        if predicted_role == role:
                            correct += 1
                            print(f"  ✓ 正确")
                        else:
                            print(f"  ✗ 错误: 期望 {role}, 得到 {predicted_role}")
                        
                        total += 1
                        time.sleep(1)  # 避免请求过快
                    else:
                        print(f"  ✗ 请求失败: {response.status_code}")
                except Exception as e:
                    print(f"  ✗ 异常: {e}")

print("=" * 60)
print(f"测试完成！")
print(f"总测试数: {total}")
print(f"正确数: {correct}")
if total > 0:
    accuracy = correct / total * 100
    print(f"准确率: {accuracy:.2f}%")
else:
    print("无测试数据")