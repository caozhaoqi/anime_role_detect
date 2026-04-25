#!/usr/bin/env python3
import requests
from PIL import Image
import io
import json

GATEWAY_URL = "http://localhost:8000"

def test_gateway_endpoints():
    print("=" * 70)
    print("API Gateway 完整测试")
    print("=" * 70)

    session = requests.Session()

    print("\n" + "=" * 70)
    print("第一部分：公共接口（无需认证）")
    print("=" * 70)

    print("\n1. GET / - 根路径")
    try:
        response = requests.get(f"{GATEWAY_URL}/", timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")

    print("\n2. GET /api/health - 健康检查")
    try:
        response = requests.get(f"{GATEWAY_URL}/api/health", timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")

    print("\n3. GET /api/services - 服务状态")
    try:
        response = requests.get(f"{GATEWAY_URL}/api/services", timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {json.dumps(response.json(), indent=4, ensure_ascii=False)}")
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")

    print("\n" + "=" * 70)
    print("第二部分：认证接口")
    print("=" * 70)

    print("\n4. POST /api/auth/login - 用户登录")
    try:
        data = {
            "username": "admin",
            "password": "admin123"
        }
        response = requests.post(
            f"{GATEWAY_URL}/api/auth/login",
            data=data,
            timeout=10
        )
        print(f"   状态码: {response.status_code}")
        result = response.json()
        print(f"   响应: {json.dumps(result, indent=4, ensure_ascii=False)}")

        access_token = None
        if result.get("success"):
            access_token = result.get("data", {}).get("access_token")
            print(f"   ✅ 登录成功，获取到 token: {access_token[:20]}..." if access_token else "   ⚠️ 未获取到 token")
        else:
            print(f"   ❌ 登录失败: {result.get('message', '未知错误')}")

    except Exception as e:
        print(f"   ❌ 请求失败: {e}")
        access_token = None

    print("\n" + "=" * 70)
    print("第三部分：需要认证的接口")
    print("=" * 70)

    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    print("\n5. POST /api/classify - 图片分类（需要认证）")
    try:
        test_image = Image.new('RGB', (224, 224), color='red')
        image_content = io.BytesIO()
        test_image.save(image_content, format='JPEG')
        image_content.seek(0)

        files = {'file': ('test.jpg', image_content.getvalue(), 'image/jpeg')}
        data_form = {
            'model_name': 'resnet18_loli8',
            'use_coreml': 'false',
            'use_model': 'true',
            'use_attributes': 'true',
            'cache_bypass': 'false',
            'multi_role': 'false',
            'use_deepdanbooru': 'true'
        }
        response = requests.post(
            f"{GATEWAY_URL}/api/classify",
            files=files,
            data=data_form,
            headers=headers,
            timeout=120
        )
        print(f"   状态码: {response.status_code}")
        result = response.json()
        print(f"   响应: {json.dumps(result, indent=4, ensure_ascii=False)}")
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")

    print("\n6. GET /api/history - 历史记录（需要认证）")
    try:
        response = requests.get(
            f"{GATEWAY_URL}/api/history",
            headers=headers,
            timeout=10
        )
        print(f"   状态码: {response.status_code}")
        result = response.json()
        print(f"   响应: {json.dumps(result, indent=4, ensure_ascii=False)}")
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")

    print("\n" + "=" * 70)
    print("第四部分：模型服务接口（8888端口）")
    print("=" * 70)

    print("\n7. GET /api/model/health - 模型服务健康检查")
    try:
        response = requests.get(f"{GATEWAY_URL}/api/model/health", timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_gateway_endpoints()