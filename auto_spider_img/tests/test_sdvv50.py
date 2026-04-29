#!/usr/bin/env python3
"""
测试 sd.vv50.de 页面结构
"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# 配置 Chrome 选项
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# 启动浏览器
driver = webdriver.Chrome(options=chrome_options)
driver.set_page_load_timeout(30)

# 访问搜索页面
search_url = "https://sd.vv50.de/search?q=アロナ"
driver.get(search_url)
print(f"已访问: {search_url}")

# 等待页面加载
time.sleep(5)

# 获取页面标题
print(f"页面标题: {driver.title}")

# 获取页面源代码
page_source = driver.page_source
print(f"\n页面源代码长度: {len(page_source)}")

# 解析页面
soup = BeautifulSoup(page_source, 'html.parser')

# 查找所有图片
img_tags = soup.find_all('img')
print(f"\n找到 {len(img_tags)} 个 <img> 标签")

for i, img in enumerate(img_tags[:10]):
    src = img.get('src') or img.get('data-src') or img.get('data-original')
    print(f"  图片 {i+1}: {src}")

# 查找所有链接
links = soup.find_all('a')
print(f"\n找到 {len(links)} 个 <a> 标签")

for i, link in enumerate(links[:10]):
    href = link.get('href')
    text = link.get_text(strip=True)
    print(f"  链接 {i+1}: {href} - {text[:50]}")

# 查找所有 div
divs = soup.find_all('div')
print(f"\n找到 {len(divs)} 个 <div> 标签")

# 查找可能的图片容器
possible_containers = soup.find_all(['div', 'article', 'figure', 'section'])
print(f"\n找到 {len(possible_containers)} 个可能的图片容器")

# 保存页面源代码以便分析
with open('sdvv50_page.html', 'w', encoding='utf-8') as f:
    f.write(page_source)
print("\n页面源代码已保存到 sdvv50_page.html")

# 关闭浏览器
driver.quit()
print("\n浏览器已关闭")
