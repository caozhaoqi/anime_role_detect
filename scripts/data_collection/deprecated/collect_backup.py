#!/usr/bin/env python3
"""
备用数据源采集脚本
尝试多个可能可用的数据源
"""
import os
import sys
import time
import json
import requests
import hashlib
import random
from PIL import Image
from io import BytesIO
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collect_backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupSourceCollector:
    """备用数据源采集器"""

    def __init__(self, output_dir='./backup_images'):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })
        os.makedirs(output_dir, exist_ok=True)

    def try_picsum_photos(self):
        """尝试Picsum Photos"""
        try:
            url = f"https://picsum.photos/800/800"
            response = self.session.get(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                if len(response.content) > 5000:
                    return response.content
        except Exception as e:
            logger.debug(f"Picsum失败: {e}")
        return None

    def try_dogceo(self):
        """尝试Dog CEO（动物图片）"""
        try:
            url = "https://dog.ceo/api/breeds/image/random"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    img_url = data.get('message')
                    img_response = self.session.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        return img_response.content
        except Exception as e:
            logger.debug(f"Dog CEO失败: {e}")
        return None

    def try_cat_as_a_service(self):
        """尝试Cat as a Service"""
        try:
            url = "https://cataas.com/cat?json=true"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'url' in data:
                    img_url = f"https://cataas.com{data['url']}"
                    img_response = self.session.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        return img_response.content
        except Exception as e:
            logger.debug(f"Cat as a Service失败: {e}")
        return None

    def try_nekos_life(self):
        """尝试Nekos Life"""
        endpoints = ['https://api.nekos.life/v2/img/neko', 'https://api.nekos.life/v2/img/sfw/neko']
        for url in endpoints:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'url' in data:
                        img_response = self.session.get(data['url'], timeout=10)
                        if img_response.status_code == 200:
                            return img_response.content
            except:
                pass
            time.sleep(0.3)
        return None

    def try_shibe_online(self):
        """尝试Shibe Online"""
        try:
            url = "http://shibe.online/api/shibes?count=1&urls=true&httpsUrls=true"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    img_url = data[0]
                    img_response = self.session.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        return img_response.content
        except Exception as e:
            logger.debug(f"Shibe失败: {e}")
        return None

    def try_thispersondoesnotexist(self):
        """尝试This Person Does Not Exist（AI生成人脸）"""
        try:
            url = "https://thispersondoesnotexist.com/image"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                if len(response.content) > 5000:
                    return response.content
        except Exception as e:
            logger.debug(f"ThisPerson失败: {e}")
        return None

    def try_source_unsplash(self):
        """尝试Unsplash随机图片"""
        try:
            url = "https://source.unsplash.com/random/800x800"
            response = self.session.get(url, timeout=15, allow_redirects=True)
            if response.status_code == 200:
                if len(response.content) > 5000:
                    return response.content
        except Exception as e:
            logger.debug(f"Unsplash Source失败: {e}")
        return None

    def try_pravatar(self):
        """尝试Pravatar（头像图片）"""
        try:
            url = f"https://i.pravatar.cc/800?img={random.randint(1, 70)}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                if len(response.content) > 5000:
                    return response.content
        except Exception as e:
            logger.debug(f"Pravatar失败: {e}")
        return None

    def try_xkcd(self):
        """尝试XKCD漫画"""
        try:
            url = "https://xkcd.com/info.0.json"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'img' in data:
                    img_response = self.session.get(data['img'], timeout=10)
                    if img_response.status_code == 200:
                        return img_response.content
        except Exception as e:
            logger.debug(f"XKCD失败: {e}")
        return None

    def try_all_sources(self):
        """尝试所有备用数据源"""
        sources = [
            self.try_source_unsplash,
            self.try_picsum_photos,
            self.try_nekos_life,
            self.try_dogceo,
            self.try_cat_as_a_service,
            self.try_shibe_online,
            self.try_thispersondoesnotexist,
            self.try_pravatar,
            self.try_xkcd,
        ]

        for source in sources:
            try:
                content = source()
                if content and len(content) > 5000:
                    return content
            except:
                pass
            time.sleep(0.3)

        return None

    def is_valid_image(self, content):
        """检查图片是否有效"""
        try:
            img = Image.open(BytesIO(content))
            width, height = img.size
            if width >= 512 and height >= 512:
                return True, width, height
            return False, width, height
        except:
            return False, 0, 0

    def collect_for_role(self, role_name, target_count=30):
        """为角色采集图片"""
        role_dir = os.path.join(self.output_dir, role_name)
        os.makedirs(role_dir, exist_ok=True)

        existing = [f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.png'))]
        existing_count = len(existing)

        if existing_count >= target_count:
            logger.info(f"角色 {role_name} 已有 {existing_count} 张图片，跳过")
            return {'role': role_name, 'success': 0, 'failed': 0, 'existing': existing_count}

        need_count = target_count - existing_count
        logger.info(f"角色 {role_name} 需要采集 {need_count} 张图片")

        success_count = 0
        failed_count = 0

        for i in range(need_count * 3):
            if success_count >= need_count:
                break

            content = self.try_all_sources()

            if not content:
                failed_count += 1
                continue

            is_valid, width, height = self.is_valid_image(content)
            if not is_valid:
                failed_count += 1
                continue

            file_hash = hashlib.md5(content).hexdigest()
            save_path = os.path.join(role_dir, f"{file_hash}.jpg")

            if os.path.exists(save_path):
                continue

            try:
                with open(save_path, 'wb') as f:
                    f.write(content)
                success_count += 1
                logger.debug(f"成功: {role_name} - {width}x{height}")
            except Exception as e:
                failed_count += 1
                logger.debug(f"保存失败: {e}")

            time.sleep(0.5)

        logger.info(f"角色 {role_name} 采集完成: 成功 {success_count}, 失败 {failed_count}")
        return {
            'role': role_name,
            'success': success_count,
            'failed': failed_count,
            'existing': existing_count
        }

    def collect_all(self, role_list, target_count=30):
        """采集所有角色"""
        logger.info(f"开始采集备用数据源，共 {len(role_list)} 个角色")

        results = []
        for role in role_list:
            result = self.collect_for_role(role, target_count)
            results.append(result)

        total_success = sum(r['success'] for r in results)
        total_failed = sum(r['failed'] for r in results)

        logger.info(f"\n=== 采集完成 ===")
        logger.info(f"成功下载: {total_success}")
        logger.info(f"下载失败: {total_failed}")

        return {
            'total_roles': len(results),
            'total_success': total_success,
            'total_failed': total_failed,
            'details': results
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='备用数据源采集')
    parser.add_argument('--role_file', type=str, required=True, help='角色列表文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--target_count', type=int, default=30, help='目标数量')

    args = parser.parse_args()

    with open(args.role_file, 'r', encoding='utf-8') as f:
        role_list = [line.strip() for line in f if line.strip()]

    logger.info(f"读取到 {len(role_list)} 个角色")

    collector = BackupSourceCollector(output_dir=args.output_dir)
    result = collector.collect_all(role_list, args.target_count)

    with open('backup_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n采集完成！")

if __name__ == "__main__":
    main()
