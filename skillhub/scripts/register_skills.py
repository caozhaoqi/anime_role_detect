#!/usr/bin/env python3
import requests
import json

SKILLS = [
    {
        "id": "ardc-cleaner",
        "name": "数据清洗器",
        "version": "1.0.0",
        "description": "数据清洗技能，用于清洗和预处理动漫图片数据。支持图片格式转换、尺寸调整、颜色校正、噪声去除。",
        "author": "ARD Team",
        "category": "cleaner",
        "status": "stable",
        "entry_point": "scripts/clean_images.py",
        "tags": ["数据清洗", "图片处理", "预处理"],
        "runtime": "python"
    },
    {
        "id": "ardc-classifier",
        "name": "角色分类器",
        "version": "1.0.0",
        "description": "角色分类识别技能，使用深度学习模型识别动漫角色。支持多分类、特征提取、相似度匹配。",
        "author": "ARD Team",
        "category": "classifier",
        "status": "testing",
        "entry_point": "scripts/classify.py",
        "tags": ["分类", "AI", "深度学习"],
        "runtime": "python"
    },
    {
        "id": "ardc-trainer",
        "name": "模型训练器",
        "version": "1.0.0",
        "description": "模型训练技能，用于训练和优化角色检测模型。支持数据集管理、模型训练、超参数调优、模型评估。",
        "author": "ARD Team",
        "category": "trainer",
        "status": "development",
        "entry_point": "scripts/train.py",
        "tags": ["训练", "AI", "模型"],
        "runtime": "python"
    },
    {
        "id": "ardc-search",
        "name": "图片搜索器",
        "version": "1.0.0",
        "description": "图片搜索检索技能，支持基于内容的图像检索。支持相似度搜索、反向图片搜索、图像聚类。",
        "author": "ARD Team",
        "category": "search",
        "status": "stable",
        "entry_point": "scripts/search.py",
        "tags": ["搜索", "图像检索", "CBIR"],
        "runtime": "python"
    },
    {
        "id": "ardc-analyzer",
        "name": "数据分析器",
        "version": "1.0.0",
        "description": "数据分析技能，用于分析和可视化动漫数据。支持统计分析、趋势分析、数据可视化、报告生成。",
        "author": "ARD Team",
        "category": "analyzer",
        "status": "stable",
        "entry_point": "scripts/analyze.py",
        "tags": ["分析", "可视化", "统计"],
        "runtime": "python"
    },
    {
        "id": "ardc-exporter",
        "name": "数据导出器",
        "version": "1.0.0",
        "description": "数据导出技能，支持多种格式的数据导出。支持CSV、JSON、Excel、SQLite等格式导出。",
        "author": "ARD Team",
        "category": "utility",
        "status": "stable",
        "entry_point": "scripts/export.py",
        "tags": ["导出", "数据", "格式转换"],
        "runtime": "python"
    },
    {
        "id": "ardc-importer",
        "name": "数据导入器",
        "version": "1.0.0",
        "description": "数据导入技能，支持从多种数据源导入数据。支持CSV、JSON、API、数据库导入。",
        "author": "ARD Team",
        "category": "utility",
        "status": "stable",
        "entry_point": "scripts/import.py",
        "tags": ["导入", "数据", "ETL"],
        "runtime": "python"
    },
    {
        "id": "ardc-validator",
        "name": "数据验证器",
        "version": "1.0.0",
        "description": "数据验证技能，用于验证数据质量和完整性。支持格式验证、规则检查、数据一致性验证。",
        "author": "ARD Team",
        "category": "cleaner",
        "status": "testing",
        "entry_point": "scripts/validate.py",
        "tags": ["验证", "数据质量", "检查"],
        "runtime": "python"
    },
    {
        "id": "ardc-transformer",
        "name": "数据转换器",
        "version": "1.0.0",
        "description": "数据转换技能，用于数据格式转换和映射。支持字段映射、类型转换、数据标准化。",
        "author": "ARD Team",
        "category": "cleaner",
        "status": "stable",
        "entry_point": "scripts/transform.py",
        "tags": ["转换", "映射", "标准化"],
        "runtime": "python"
    },
    {
        "id": "ardc-notifier",
        "name": "消息通知器",
        "version": "1.0.0",
        "description": "消息通知技能，支持多种通知渠道。支持邮件、钉钉、微信、短信通知。",
        "author": "ARD Team",
        "category": "utility",
        "status": "stable",
        "entry_point": "scripts/notify.py",
        "tags": ["通知", "消息", "告警"],
        "runtime": "python"
    }
]

def main():
    base_url = "http://localhost:8000/api/skills"
    
    print("开始注册技能...")
    for skill in SKILLS:
        try:
            response = requests.post(base_url, json=skill)
            if response.status_code == 200:
                print(f"✓ 注册成功: {skill['name']} ({skill['id']})")
            else:
                print(f"✗ 注册失败: {skill['name']} - {response.text}")
        except Exception as e:
            print(f"✗ 注册异常: {skill['name']} - {str(e)}")
    
    print("\n注册完成！")
    
    # 获取当前技能列表
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            data = response.json()
            print(f"\n当前技能总数: {len(data['skills'])}")
            for s in data['skills']:
                print(f"  - {s['name']} ({s['id']})")
    except Exception as e:
        print(f"获取技能列表失败: {str(e)}")

if __name__ == "__main__":
    main()
