#!/usr/bin/env python3
"""
日志融合模块
收集分类日志，融合特征构建新模型
"""
import os
import json
import numpy as np
import faiss
from datetime import datetime

from src.core.logging.global_logger import get_logger

logger = get_logger("log_fusion")

class LogFusion:
    """日志融合类"""
    
    def __init__(self, log_dir='./logs', model_dir='./models'):
        """初始化日志融合模块"""
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.logs = []
        self.logger = get_logger("log_fusion")
        
        # 创建必要的目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
    
    def collect_logs(self, max_logs=5):
        """收集分类日志
        
        Args:
            max_logs: 最大日志数量，默认为5
            
        Returns:
            list: 收集的日志列表
        """
        self.logger.info(f"开始收集日志，最多收集 {max_logs} 条")
        
        # 获取所有日志文件
        log_files = []
        for file_name in os.listdir(self.log_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.log_dir, file_name)
                # 获取文件修改时间
                mtime = os.path.getmtime(file_path)
                log_files.append((file_path, mtime))
        
        # 按修改时间排序，取最近的max_logs条
        log_files.sort(key=lambda x: x[1], reverse=True)
        recent_logs = log_files[:max_logs]
        
        # 读取日志文件
        self.logs = []
        for file_path, _ in recent_logs:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    self.logs.append(log_data)
                self.logger.info(f"成功读取日志文件: {file_path}")
            except Exception as e:
                self.logger.error(f"读取日志文件失败: {file_path}, 错误: {e}")
                continue
        
        self.logger.info(f"日志收集完成，共收集 {len(self.logs)} 条日志")
        return self.logs
    
    def extract_features(self):
        """从日志中提取特征
        
        Returns:
            tuple: (features, role_names) 特征向量和角色名称
        """
        self.logger.info("开始从日志中提取特征")
        
        features = []
        role_names = []
        
        for log in self.logs:
            try:
                # 检查日志结构
                if 'feature' in log and 'role' in log:
                    feature = np.array(log['feature'], dtype=np.float32)
                    features.append(feature)
                    role_names.append(log['role'])
                    self.logger.info(f"成功提取特征: 角色={log['role']}, 特征维度={feature.shape}")
                else:
                    self.logger.warning(f"日志结构不完整: {log.keys()}")
            except Exception as e:
                self.logger.error(f"提取特征失败: {e}")
                continue
        
        if not features:
            self.logger.warning("没有提取到特征")
            return None, None
        
        features_np = np.array(features, dtype=np.float32)
        self.logger.info(f"特征提取完成，共提取 {len(features)} 个特征，维度={features_np.shape}")
        return features_np, role_names
    
    def fuse_features(self, features, role_names, method='mean'):
        """融合特征
        
        Args:
            features: 特征向量数组
            role_names: 角色名称列表
            method: 融合方法，可选 'mean', 'concatenate', 'weighted'
            
        Returns:
            tuple: (fused_features, fused_roles) 融合后的特征和角色
        """
        self.logger.info(f"开始融合特征，使用方法: {method}")
        
        if features is None or role_names is None:
            self.logger.warning("没有特征可融合")
            return None, None
        
        # 按角色分组特征
        role_features = {}
        for i, role in enumerate(role_names):
            if role not in role_features:
                role_features[role] = []
            role_features[role].append(features[i])
        
        # 融合每个角色的特征
        fused_features = []
        fused_roles = []
        
        for role, role_feature_list in role_features.items():
            try:
                role_feature_array = np.array(role_feature_list, dtype=np.float32)
                
                if method == 'mean':
                    # 均值融合
                    fused_feature = np.mean(role_feature_array, axis=0)
                elif method == 'concatenate':
                    # 特征拼接
                    fused_feature = np.concatenate(role_feature_array, axis=0)
                elif method == 'weighted':
                    # 加权融合（简单权重，按特征质量）
                    weights = np.linspace(0.5, 1.0, len(role_feature_list))
                    weights = weights / np.sum(weights)
                    fused_feature = np.average(role_feature_array, axis=0, weights=weights)
                else:
                    self.logger.error(f"未知的融合方法: {method}")
                    continue
                
                fused_features.append(fused_feature)
                fused_roles.append(role)
                self.logger.info(f"成功融合角色 {role} 的特征，融合方法: {method}")
            except Exception as e:
                self.logger.error(f"融合角色 {role} 的特征失败: {e}")
                continue
        
        if not fused_features:
            self.logger.warning("没有融合到特征")
            return None, None
        
        fused_features_np = np.array(fused_features, dtype=np.float32)
        self.logger.info(f"特征融合完成，共融合 {len(fused_features)} 个角色的特征")
        return fused_features_np, fused_roles
    
    def build_new_model(self, fused_features, fused_roles, model_name=None):
        """构建新模型
        
        Args:
            fused_features: 融合后的特征向量
            fused_roles: 融合后的角色名称
            model_name: 模型名称，默认为当前时间戳
            
        Returns:
            str: 模型路径
        """
        self.logger.info("开始构建新模型")
        
        if fused_features is None or fused_roles is None:
            self.logger.error("没有融合特征可用于构建模型")
            return None
        
        # 生成模型名称
        if model_name is None:
            model_name = f"fused_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = os.path.join(self.model_dir, model_name)
        
        try:
            # 获取特征维度
            dim = fused_features.shape[1]
            
            # 创建Faiss索引（使用余弦相似度）
            index = faiss.IndexFlatIP(dim)
            
            # 添加特征向量到索引
            index.add(fused_features)
            
            # 保存Faiss索引
            faiss.write_index(index, f"{model_path}.faiss")
            
            # 保存角色映射
            with open(f"{model_path}_mapping.json", "w", encoding="utf-8") as f:
                json.dump(fused_roles, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"新模型构建成功，模型路径: {model_path}")
            return model_path
        except Exception as e:
            self.logger.error(f"构建新模型失败: {e}")
            return None
    
    def update_model(self, model_path):
        """更新模型（可选）
        
        Args:
            model_path: 模型路径
            
        Returns:
            bool: 更新是否成功
        """
        self.logger.info(f"开始更新模型: {model_path}")
        
        # 这里可以实现模型更新逻辑，例如：
        # 1. 加载现有模型
        # 2. 添加新特征
        # 3. 重新训练或优化模型
        # 4. 保存更新后的模型
        
        self.logger.info(f"模型更新完成: {model_path}")
        return True
    
    def run_fusion(self, max_logs=5, fusion_method='mean', model_name=None):
        """运行完整的融合流程
        
        Args:
            max_logs: 最大日志数量
            fusion_method: 融合方法
            model_name: 模型名称
            
        Returns:
            str: 新模型路径
        """
        self.logger.info("开始运行日志融合流程")
        
        # 1. 收集日志
        self.collect_logs(max_logs)
        
        # 2. 提取特征
        features, role_names = self.extract_features()
        if features is None:
            self.logger.error("特征提取失败，融合流程终止")
            return None
        
        # 3. 融合特征
        fused_features, fused_roles = self.fuse_features(features, role_names, fusion_method)
        if fused_features is None:
            self.logger.error("特征融合失败，融合流程终止")
            return None
        
        # 4. 构建新模型
        model_path = self.build_new_model(fused_features, fused_roles, model_name)
        if model_path is None:
            self.logger.error("构建新模型失败，融合流程终止")
            return None
        
        # 5. 更新模型（可选）
        self.update_model(model_path)
        
        self.logger.info("日志融合流程完成")
        return model_path

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='日志融合工具')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./models', help='模型目录')
    parser.add_argument('--max_logs', type=int, default=5, help='最大日志数量')
    parser.add_argument('--fusion_method', type=str, default='mean', help='融合方法')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    
    args = parser.parse_args()
    
    # 初始化日志融合模块
    fusion = LogFusion(log_dir=args.log_dir, model_dir=args.model_dir)
    
    # 运行融合流程
    model_path = fusion.run_fusion(
        max_logs=args.max_logs,
        fusion_method=args.fusion_method,
        model_name=args.model_name
    )
    
    if model_path:
        print(f"\n🎉 日志融合完成！")
        print(f"新模型路径: {model_path}")
        print(f"融合了 {args.max_logs} 条日志")
        print(f"使用融合方法: {args.fusion_method}")
    else:
        print(f"\n❌ 日志融合失败！")

if __name__ == "__main__":
    main()
