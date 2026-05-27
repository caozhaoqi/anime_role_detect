#!/usr/bin/env python3
import requests
import json

SKILLS = [
    # ==================== 数据采集层技能 ====================
    {
        "id": "ardc-collector",
        "name": "数据采集器",
        "version": "2.0.1",
        "description": "数据采集技能，用于采集动漫角色图片数据。支持URL爬取、图片下载、批量采集、补充采集。",
        "author": "ARD Team",
        "category": "data",
        "status": "stable",
        "entry_point": "scripts/data_collection/download_images.py",
        "tags": ["数据采集", "图片下载", "爬虫", "批量"],
        "runtime": "python",
        "dependencies": []
    },
    {
        "id": "ardc-spider",
        "name": "爬虫服务",
        "version": "2.0.0",
        "description": "单角色爬虫技能，通过API爬取指定角色的图片URL。支持异步爬取、结果缓存、多源采集。",
        "author": "ARD Team",
        "category": "data",
        "status": "stable",
        "entry_point": "scripts/data_collection/spider_single_role.py",
        "tags": ["爬虫", "URL采集", "单角色", "API"],
        "runtime": "python",
        "dependencies": ["ardc-collector"]
    },
    {
        "id": "ardc-supplement",
        "name": "数据补充器",
        "version": "1.1.0",
        "description": "数据补充技能，自动检测并补充图片数量不足的角色。支持智能补充、优先级调度、增量采集。",
        "author": "ARD Team",
        "category": "data",
        "status": "stable",
        "entry_point": "scripts/data_collection/supplement_low_count_roles.py",
        "tags": ["数据补充", "增量采集", "智能调度"],
        "runtime": "python",
        "dependencies": ["ardc-collector", "ardc-spider"]
    },
    {
        "id": "ardc-organizer",
        "name": "数据集组织器",
        "version": "1.0.0",
        "description": "数据集组织技能，将下载的图片整理到final_dataset目录。支持格式统一、角色分类、目录同步。",
        "author": "ARD Team",
        "category": "data",
        "status": "stable",
        "entry_point": "scripts/data_collection/organize_final_dataset.py",
        "tags": ["数据集", "组织", "格式统一", "目录同步"],
        "runtime": "python",
        "dependencies": ["ardc-collector"]
    },
    # ==================== 数据清洗层技能 ====================
    {
        "id": "ardc-cleaner",
        "name": "数据清洗器",
        "version": "2.1.0",
        "description": "数据清洗技能，用于清洗和预处理动漫图片数据。支持图片格式转换、去重、低质量数据过滤、质量评估。",
        "author": "ARD Team",
        "category": "data",
        "status": "stable",
        "entry_point": "scripts/data_cleaning/clean_final_dataset.py",
        "tags": ["数据清洗", "图片处理", "去重", "质量评估"],
        "runtime": "python",
        "dependencies": ["ardc-organizer"]
    },
    {
        "id": "ardc-deduplicator",
        "name": "重复检测工具",
        "version": "1.0.0",
        "description": "重复图片检测技能，基于内容哈希去重。支持批量去重、相似度匹配、重复组管理。",
        "author": "ARD Team",
        "category": "data",
        "status": "stable",
        "entry_point": "scripts/data_cleaning/archived/clean_duplicates.py",
        "tags": ["去重", "哈希", "相似度", "图片处理"],
        "runtime": "python",
        "dependencies": []
    },
    {
        "id": "ardc-tagger",
        "name": "标签标注器",
        "version": "1.0.0",
        "description": "DeepDanbooru标签标注技能，自动为图片添加内容标签。支持批量标注、标签过滤、角色识别。",
        "author": "ARD Team",
        "category": "data",
        "status": "testing",
        "entry_point": "scripts/data_cleaning/deepdanbooru_tagger.py",
        "tags": ["标签", "标注", "DeepDanbooru", "内容识别"],
        "runtime": "python",
        "dependencies": ["ardc-cleaner"]
    },
    {
        "id": "ardc-balance",
        "name": "数据集平衡器",
        "version": "1.0.0",
        "description": "数据集平衡技能，调整各角色图片数量至均衡。支持填充至目标数量、数据增强、采样策略。",
        "author": "ARD Team",
        "category": "data",
        "status": "stable",
        "entry_point": "scripts/data_cleaning/balance_dataset.py",
        "tags": ["数据平衡", "采样", "数据增强", "均衡"],
        "runtime": "python",
        "dependencies": ["ardc-cleaner"]
    },
    # ==================== AI模型层技能 ====================
    {
        "id": "ardc-trainer",
        "name": "模型训练器",
        "version": "2.0.0",
        "description": "模型训练技能，用于训练和优化角色检测模型。支持数据集管理、模型训练、超参数调优、模型评估。",
        "author": "ARD Team",
        "category": "ai",
        "status": "stable",
        "entry_point": "scripts/model_training/train_loli_optimized.py",
        "tags": ["训练", "AI", "模型", "评估"],
        "runtime": "python",
        "dependencies": ["ardc-cleaner"]
    },
    {
        "id": "ardc-classifier",
        "name": "角色分类器",
        "version": "2.0.0",
        "description": "角色分类识别技能，使用深度学习模型识别动漫角色。支持多分类、特征提取、相似度匹配。",
        "author": "ARD Team",
        "category": "ai",
        "status": "stable",
        "entry_point": "scripts/classification/classify_collection_local.py",
        "tags": ["分类", "AI", "深度学习", "识别"],
        "runtime": "python",
        "dependencies": ["ardc-trainer"]
    },
    {
        "id": "ardc-evaluator",
        "name": "模型评估器",
        "version": "1.0.0",
        "description": "模型评估技能，用于评估模型性能。支持准确率分析、混淆矩阵、基准测试。",
        "author": "ARD Team",
        "category": "ai",
        "status": "stable",
        "entry_point": "scripts/model_evaluation/evaluate_model.py",
        "tags": ["评估", "AI", "基准测试", "准确率"],
        "runtime": "python",
        "dependencies": ["ardc-classifier"]
    },
    {
        "id": "ardc-detector",
        "name": "人脸检测器",
        "version": "1.0.0",
        "description": "人脸检测技能，使用YuNet模型检测动漫人脸。支持批量检测、关键点定位、检测优化。",
        "author": "ARD Team",
        "category": "ai",
        "status": "stable",
        "entry_point": "scripts/detection/run_yunet_detection_optimized.py",
        "tags": ["人脸检测", "YuNet", "关键点", "动漫"],
        "runtime": "python",
        "dependencies": []
    },
    # ==================== 分析报告层技能 ====================
    {
        "id": "ardc-analyzer",
        "name": "数据分析器",
        "version": "1.0.0",
        "description": "数据分析技能，用于分析和可视化动漫数据。支持统计分析、趋势分析、数据可视化、报告生成。",
        "author": "ARD Team",
        "category": "utility",
        "status": "stable",
        "entry_point": "scripts/analysis/analyze_feature_quality.py",
        "tags": ["分析", "可视化", "统计", "报告"],
        "runtime": "python",
        "dependencies": []
    },
    {
        "id": "ardc-index",
        "name": "FAISS索引构建器",
        "version": "1.0.0",
        "description": "向量索引构建技能，使用FAISS构建图片特征索引。支持快速搜索、相似度匹配、索引优化。",
        "author": "ARD Team",
        "category": "utility",
        "status": "stable",
        "entry_point": "scripts/analysis/build_faiss_index.py",
        "tags": ["索引", "FAISS", "搜索", "向量"],
        "runtime": "python",
        "dependencies": ["ardc-classifier"]
    },
    {
        "id": "ardc-reporter",
        "name": "报告生成器",
        "version": "1.0.0",
        "description": "报告生成技能，生成数据集统计报告和质量分析报告。支持角色分布、质量评估、统计摘要。",
        "author": "ARD Team",
        "category": "utility",
        "status": "stable",
        "entry_point": "scripts/analysis/generate_final_report.py",
        "tags": ["报告", "统计", "质量分析", "数据集"],
        "runtime": "python",
        "dependencies": ["ardc-analyzer"]
    },
    # ==================== API服务层技能 ====================
    {
        "id": "ardc-api",
        "name": "API服务",
        "version": "1.0.0",
        "description": "飞书命令服务技能，提供飞书机器人命令接口。支持分类查询、状态查询、命令执行。",
        "author": "ARD Team",
        "category": "service",
        "status": "stable",
        "entry_point": "scripts/api/feishu_command_server.py",
        "tags": ["API", "飞书", "机器人", "命令"],
        "runtime": "python",
        "dependencies": ["ardc-classifier"]
    },
    {
        "id": "ardc-skillhub",
        "name": "技能中心服务",
        "version": "2.0.0",
        "description": "技能管理服务，提供技能注册、查询、更新和删除功能。支持技能分类、版本管理、依赖管理。",
        "author": "ARD Team",
        "category": "service",
        "status": "stable",
        "entry_point": "scripts/skillhub_server.py",
        "tags": ["技能管理", "API", "注册", "服务"],
        "runtime": "python",
        "dependencies": []
    },
    {
        "id": "ardc-cli",
        "name": "命令行工具",
        "version": "1.0.0",
        "description": "命令行接口技能，提供终端命令行交互能力。支持命令执行、状态查询、批量操作。",
        "author": "ARD Team",
        "category": "service",
        "status": "stable",
        "entry_point": "scripts/ardc-cli.py",
        "tags": ["CLI", "命令行", "终端", "工具"],
        "runtime": "python",
        "dependencies": []
    },
    {
        "id": "ardc-notifier",
        "name": "通知服务",
        "version": "1.0.0",
        "description": "通知服务技能，提供消息通知和状态推送能力。支持飞书通知、邮件通知、日志推送。",
        "author": "ARD Team",
        "category": "service",
        "status": "stable",
        "entry_point": "scripts/common/notification_utils.py",
        "tags": ["通知", "消息", "推送", "飞书"],
        "runtime": "python",
        "dependencies": []
    },
    {
        "id": "ardc-importer",
        "name": "数据导入器",
        "version": "1.0.0",
        "description": "数据导入技能，用于导入外部数据源到数据库。支持URL导入、批量导入、数据同步。",
        "author": "ARD Team",
        "category": "data",
        "status": "stable",
        "entry_point": "scripts/data_collection/database/import_all_urls.py",
        "tags": ["导入", "数据同步", "数据库", "批量"],
        "runtime": "python",
        "dependencies": ["ardc-collector"]
    }
]

def main():
    base_url = "http://localhost:8000/api/skills"
    
    print("开始注册技能...")
    print("=" * 70)
    
    for skill in SKILLS:
        try:
            response = requests.post(base_url, json=skill)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✓ 注册成功: {skill['name']} ({skill['id']}) v{skill['version']}")
                else:
                    print(f"✓ 已存在: {skill['name']} ({skill['id']}) - {result.get('message')}")
            else:
                print(f"✗ 注册失败: {skill['name']} - {response.text}")
        except Exception as e:
            print(f"✗ 注册异常: {skill['name']} - {str(e)}")
    
    print("=" * 70)
    print("\n注册完成！")
    
    # 获取当前技能列表
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            data = response.json()
            print(f"\n当前技能总数: {len(data['skills'])}")
            
            # 按分类统计
            by_category = {}
            for s in data['skills']:
                cat = s.get('category', 'other')
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(s)
            
            print("\n技能分类统计:")
            for cat, skills in sorted(by_category.items()):
                print(f"  {cat}: {len(skills)} 个")
                for s in sorted(skills, key=lambda x: x['name']):
                    print(f"    - {s['name']} ({s['id']}) v{s['version']}")
    except Exception as e:
        print(f"获取技能列表失败: {str(e)}")

if __name__ == "__main__":
    main()
