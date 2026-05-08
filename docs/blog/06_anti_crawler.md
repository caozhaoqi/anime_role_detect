# 【技术难点】爬虫反爬机制突破

> 在数据采集过程中，目标网站的反爬机制是最大的技术障碍之一。

---

## 🔍 问题背景

目标网站采用多种反爬策略：

| 反爬手段 | 表现形式 | 影响 |
|---------|---------|------|
| User-Agent 检测 | 返回 403/404 | 请求被拒绝 |
| Referer 验证 | 图片下载失败 | 无法获取资源 |
| 频率限制 | 请求被限流 | 采集速度受限 |
| Cloudflare 验证 | 需要验证码 | 无法正常访问 |
| IP 封禁 | 访问被阻断 | 彻底无法访问 |

---

## 💡 解决方案：反爬策略

### 请求头伪装

```python
import requests
from fake_useragent import UserAgent

class AntiCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.delay = 1.5
    
    def get_headers(self):
        """生成随机请求头"""
        return {
            'User-Agent': self.ua.random,
            'Referer': 'https://www.pixiv.net/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
        }
    
    def safe_request(self, url, max_retries=3):
        """安全请求，带重试机制"""
        for attempt in range(max_retries):
            try:
                headers = self.get_headers()
                response = self.session.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    return response
                
                if response.status_code == 403:
                    print(f"⚠️ 请求被拒绝，等待 {self.delay * (attempt+1)} 秒")
                    time.sleep(self.delay * (attempt + 1))
                    continue
                    
            except Exception as e:
                print(f"❌ 请求异常: {e}")
                time.sleep(self.delay)
        
        return None
```

### 图片下载优化

```python
import hashlib
import os

def download_image(url, save_dir, max_retries=3):
    """下载图片，支持重试和去重"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.pixiv.net/',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    }
    
    # 生成唯一文件名（去重）
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = '.jpg' if '.jpg' in url.lower() else '.png'
    save_path = os.path.join(save_dir, f"{url_hash}{ext}")
    
    # 检查是否已存在
    if os.path.exists(save_path):
        return True
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            
            elif response.status_code == 403:
                time.sleep(2)
                
        except Exception as e:
            time.sleep(2)
    
    return False
```

---

## 🚀 使用示例

```python
# 初始化爬虫
crawler = AntiCrawler()

# 获取页面内容
url = "https://example.com/artworks/12345"
response = crawler.safe_request(url)
if response:
    print("✅ 页面获取成功")
    
    # 解析页面提取图片URL
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img', class_='original-image')
    
    # 下载图片
    save_dir = "data/images"
    os.makedirs(save_dir, exist_ok=True)
    
    for img in img_tags:
        img_url = img.get('src')
        if img_url:
            success = download_image(img_url, save_dir)
            print(f"下载 {img_url}: {'成功' if success else '失败'}")
```

---

## ⚡ 进阶策略

### 代理轮换

```python
class ProxyManager:
    def __init__(self, proxies=None):
        self.proxies = proxies or []
        self.current_index = 0
    
    def get_proxy(self):
        """获取下一个代理"""
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.proxies)
        return proxy

# 使用示例
proxies = [
    {"http": "http://proxy1:8080"},
    {"http": "http://proxy2:8080"},
    {"http": "http://proxy3:8080"}
]

manager = ProxyManager(proxies)
proxy = manager.get_proxy()
response = requests.get(url, proxies=proxy)
```

### 请求频率控制

```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_timestamps = defaultdict(list)
    
    def wait(self, key="default"):
        """等待直到可以发送下一个请求"""
        now = time.time()
        
        # 清理过期的时间戳
        self.request_timestamps[key] = [
            t for t in self.request_timestamps[key]
            if now - t < self.time_window
        ]
        
        # 如果请求数超过限制，等待
        if len(self.request_timestamps[key]) >= self.max_requests:
            wait_time = self.time_window - (now - self.request_timestamps[key][0])
            if wait_time > 0:
                time.sleep(wait_time)
        
        # 记录当前请求时间
        self.request_timestamps[key].append(time.time())

# 使用示例
limiter = RateLimiter(max_requests=10, time_window=60)

for url in urls:
    limiter.wait("example.com")
    response = requests.get(url)
```

---

## 📝 关键要点

1. **请求头伪装**：使用随机 User-Agent 绕过检测
2. **会话保持**：使用 Session 维持连接状态
3. **指数退避**：失败时递增延迟重试
4. **代理轮换**：使用代理IP避免IP封禁
5. **频率控制**：控制请求速率避免触发限流
6. **内容去重**：使用 MD5 哈希确保文件名唯一

---

## 📚 系列文章汇总

| 文章 | 主题 | 文件 |
|------|------|------|
| 第1篇 | 多模型集成与性能优化 | `01_multi_model_management.md` |
| 第2篇 | API Gateway 设计与实现 | `02_api_gateway.md` |
| 第3篇 | 分布式服务协调 | `03_distributed_coordination.md` |
| 第4篇 | 图像预处理与特征提取 | `04_image_preprocessing.md` |
| 第5篇 | NSFW 内容过滤 | `05_nsfw_detection.md` |
| 第6篇 | 爬虫反爬机制突破 | `06_anti_crawler.md` |

---

*感谢阅读！如有问题欢迎留言讨论。*
