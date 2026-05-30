#!/usr/bin/env python3
import requests
import time
import os

API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305/sis"
URL_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/img_url"
keyword = "姬坂乃爱"
role_name = "Himesaka"

print(f"启动爬虫: {keyword}")
response = requests.post(f"{API_BASE_URL}/spider_start/single", params={"key_word": keyword})
print(f"启动响应: {response.json()}")

print("等待爬取完成...")
for _ in range(30):
    status = requests.get(f"{API_BASE_URL}/spider/status").json()["data"]
    if status["is_running"]:
        print(f'爬取中: {status.get("current_count", 0)} 个URL', end="\r")
    else:
        print(f'\n爬取完成: {status.get("current_count", 0)} 个URL')
        break
    time.sleep(2)

print("获取结果...")
response = requests.get(f"{API_BASE_URL}/spider/result", params={"keyword": keyword})
print(f"结果响应码: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"结果: {result}")
    if result.get("code") == 0:
        urls = result.get("data", {}).get("urls", [])
        print(f"获取到 {len(urls)} 个URL")
        if urls:
            os.makedirs(URL_DIR, exist_ok=True)
            with open(f"{URL_DIR}/{role_name}_img.txt", "w", encoding="utf-8") as f:
                for url in urls:
                    f.write(url + "\n")
            print(f"URL已保存到 {URL_DIR}/{role_name}_img.txt")
