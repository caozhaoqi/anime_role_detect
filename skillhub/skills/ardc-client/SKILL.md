# ARDC Client - ARD 客户端技能

## 概述

ARD 客户端技能，提供与 ARD 服务和 HCM 系统交互的能力。

## 技能信息

| 属性 | 值 |
|------|------|
| ID | ardc-client |
| 名称 | ARD 客户端 |
| 版本 | 1.0.0 |
| 作者 | ARD Team |
| 分类 | utility |
| 状态 | stable |

## 功能特性

- ARD 服务客户端，支持角色检测
- HCM 系统客户端，支持员工/部门/岗位查询
- 支持认证管理和 API 调用
- Meta 数据编解码工具

## 入口文件

```
scripts/ard_client.py
scripts/hcm_client.py
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| base_url | string | http://localhost:8000 | ARD 服务地址 |
| hcm_url | string | - | HCM 服务地址 |

## 使用示例

```python
# ARD 客户端
from ard_client import ARDClient

client = ARDClient(base_url="http://localhost:8000")
result = client.detect_role("image.jpg")

# HCM 客户端
from hcm_client import HCMClient

hcm = HCMClient(base_url="https://hcm.example.com")
hcm.login()
employees = hcm.list_employees()
```

## 依赖

- requests >= 2.28.0

## 更新日志

### v1.0.0
- 初始版本
- ARD 客户端实现
- HCM 客户端实现
- 认证和 API 调用支持
