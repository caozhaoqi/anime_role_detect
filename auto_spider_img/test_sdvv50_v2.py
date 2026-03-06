#!/usr/bin/env python3
"""
测试 sd.vv50.de 页面结构 - 版本2
使用更多选项和等待时间
"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 配置 Chrome 选项 - 不禁用 JavaScript
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
driver.set_page_load_timeout(60)

# 访问搜索页面
search_url = "https://sd.vv50.de/search?q=アロナ"
print(f"正在访问: {search_url}")

try:
    driver.get(search_url)
    print(f"页面已加载")
    
    # 等待页面加载 - 等待 body 元素出现
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    print("Body 元素已找到")
    
    # 额外等待 JavaScript 渲染
    time.sleep(10)
    
    # 获取页面标题
    print(f"页面标题: {driver.title}")
    
    # 获取页面源代码
    page_source = driver.page_source
    print(f"页面源代码长度: {len(page_source)}")
    
    # 解析页面
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # 查找所有图片
    img_tags = soup.find_all('img')
    print(f"\n找到 {len(img_tags)} 个 <img> 标签")
    
    for i, img in enumerate(img_tags[:20]):
        src = img.get('src') or img.get('data-src') or img.get('data-original')
        alt = img.get('alt', '')
        print(f"  图片 {i+1}: src={src}, alt={alt[:30]}")
    
    # 查找所有链接
    links = soup.find_all('a')
    print(f"\n找到 {len(links)} 个 <a> 标签")
    
    # 查找所有 script 标签
    scripts = soup.find_all('script')
    print(f"找到 {len(scripts)} 个 <script> 标签")
    
    # 查找特定类名或 ID 的元素
    # 尝试查找常见的图片容器
    selectors = [
        'img',
        '[class*="image"]',
        '[class*="img"]',
        '[class*="picture"]',
        '[class*="photo"]',
        'article',
        'figure',
        '.gallery',
        '.grid',
        '.container'
    ]
    
    print("\n尝试查找特定选择器:")
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            print(f"  {selector}: {len(elements)} 个元素")
        except Exception as e:
            print(f"  {selector}: 查找失败 - {e}")
    
    # 保存页面源代码
    with open('sdvv50_page_v2.html', 'w', encoding='utf-8') as f:
        f.write(page_source)
    print("\n页面源代码已保存到 sdvv50_page_v2.html")
    
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 关闭浏览器
    driver.quit()
    print("\n浏览器已关闭")
