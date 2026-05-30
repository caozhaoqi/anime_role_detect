#!/usr/bin/env python3
"""HCM Cloud Client - HCM 系统认证与 API 客户端"""

import requests
import json
import base64
from typing import Dict, Any, Optional
from urllib.parse import urljoin


class HCMClient:
    """HCM 云服务客户端"""

    def __init__(self, base_url: str, username: str = None, password: str = None):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token = None
        self.session = requests.Session()

    def login(self) -> bool:
        """登录 HCM 系统"""
        url = urljoin(self.base_url, "/api/auth/login")

        try:
            response = self.session.post(
                url, json={"username": self.username, "password": self.password}
            )
            response.raise_for_status()
            result = response.json()
            self.token = result.get("access_token")
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            return True
        except Exception as e:
            print(f"登录失败: {e}")
            return False

    def call_api(self, endpoint: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
        """调用 HCM API"""
        url = urljoin(self.base_url, endpoint)

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_employee(self, employee_id: str) -> Dict[str, Any]:
        """获取员工信息"""
        return self.call_api(f"/api/employees/{employee_id}")

    def get_department(self, dept_id: str) -> Dict[str, Any]:
        """获取部门信息"""
        return self.call_api(f"/api/departments/{dept_id}")

    def get_position(self, position_id: str) -> Dict[str, Any]:
        """获取岗位信息"""
        return self.call_api(f"/api/positions/{position_id}")

    def list_employees(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """获取员工列表"""
        params = {"page": page, "page_size": page_size}
        return self.call_api("/api/employees", params=params)

    def meta_encode(self, data: Dict[str, Any]) -> str:
        """Meta 数据编码"""
        json_str = json.dumps(data)
        return base64.b64encode(json_str.encode()).decode()

    def meta_decode(self, encoded: str) -> Dict[str, Any]:
        """Meta 数据解码"""
        decoded = base64.b64decode(encoded).decode()
        return json.loads(decoded)


if __name__ == "__main__":
    client = HCMClient(base_url="https://hcm.example.com", username="admin", password="password")

    if client.login():
        print("登录成功")
        employees = client.list_employees()
        print(f"员工数量: {employees.get('total', 0)}")
    else:
        print("登录失败")
