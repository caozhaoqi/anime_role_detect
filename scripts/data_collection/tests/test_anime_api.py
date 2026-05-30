#!/usr/bin/env python3
"""
测试二次元图片API
"""
import requests
import os


def test_waifu_pics():
    """测试waifu.pics API"""
    print("测试 waifu.pics API...")
    categories = ["waifu", "neko", "shinobu", "megumin", "bully", "cuddle", "cry", "hug"]

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    for category in categories:
        try:
            url = f"https://api.waifu.pics/sfw/{category}"
            response = session.get(url, timeout=10)
            print(f"  {category}: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if "url" in data:
                    img_url = data["url"]
                    print(f"    图片URL: {img_url[:50]}...")

                    # 尝试下载
                    img_response = session.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        print(f"    下载成功，大小: {len(img_response.content)/1024:.2f} KB")
                        # 保存测试图片
                        os.makedirs("test_anime_images", exist_ok=True)
                        with open(f"test_anime_images/{category}.jpg", "wb") as f:
                            f.write(img_response.content)
                        print(f"    已保存到 test_anime_images/{category}.jpg")
                    else:
                        print(f"    下载失败: {img_response.status_code}")
                else:
                    print(f"    无URL字段")
        except Exception as e:
            print(f"  {category}: 异常 - {e}")


def test_anime_pictures():
    """测试anime-pictures.net API"""
    print("\n测试 anime-pictures.net API...")
    try:
        url = "https://anime-pictures.net/api/v3/images/random"
        response = requests.get(url, timeout=15)
        print(f"  响应: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  返回数据: {type(data)}")
            if "images" in data:
                print(f"  图片数量: {len(data['images'])}")
                if len(data["images"]) > 0:
                    img_info = data["images"][0]
                    print(f"  图片信息: {list(img_info.keys())}")
    except Exception as e:
        print(f"  异常: {e}")


def test_local_images():
    """检查本地是否已有二次元图片"""
    print("\n检查本地二次元图片...")
    import glob

    patterns = [
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/**/*.jpg",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/**/*.jpg",
    ]

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        print(f"  {pattern}: {len(files)} 张图片")
        if files:
            print(f"    示例: {files[0]}")


if __name__ == "__main__":
    test_waifu_pics()
    test_anime_pictures()
    test_local_images()
