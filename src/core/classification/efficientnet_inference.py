import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
from collections import OrderedDict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

class EfficientNetInference:
    _instance = None
    
    def __new__(cls, model_path=None, data_dir=None, enable_optimizations=True):
        if cls._instance is None:
            cls._instance = super(EfficientNetInference, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, model_path=None, data_dir=None, enable_optimizations=True):
        if self.initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.enable_optimizations = enable_optimizations
        print(f"EfficientNet推理使用设备: {self.device}")
        print(f"启用优化: {self.enable_optimizations}")
        
        # 默认路径配置
        if model_path is None:
            # 尝试查找包含测试数据角色的模型
            base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
            
            # 优先查找包含测试数据角色的模型
            augmented_model_path = os.path.join(base_model_dir, 'augmented_training', 'mobilenet_v2', 'model_best.pth')
            if os.path.exists(augmented_model_path):
                model_path = augmented_model_path
                print(f"使用包含测试数据角色的模型: {model_path}")
            else:
                # 尝试查找其他模型
                candidates = [
                    'character_classifier_best_improved.pth',
                    'character_classifier_best_v2.pth',
                    'character_classifier_best.pth'
                ]
                for cand in candidates:
                    p = os.path.join(base_model_dir, cand)
                    if os.path.exists(p):
                        model_path = p
                        break
                
                if model_path is None:
                    raise FileNotFoundError("未找到预训练模型文件")

        if data_dir is None:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'all_characters'))

        self.model_path = model_path
        self.data_dir = data_dir
        self.classes = self._load_classes()
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.initialized = True
        
        # 性能统计
        self.inference_times = []

    def _load_classes(self):
        """加载类别映射"""
        # 1. 首先尝试从模型文件中加载class_to_idx
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                # 转换为按索引排序的类别列表
                idx_to_class = {v: k for k, v in class_to_idx.items()}
                classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                print(f"从模型文件加载了 {len(classes)} 个类别")
                return classes
        except Exception as e:
            print(f"从模型文件加载类别失败: {e}")
        
        # 2. 尝试从class_to_idx.json文件加载（针对augmented_training模型）
        class_to_idx_path = os.path.join(os.path.dirname(self.model_path), "class_to_idx.json")
        if os.path.exists(class_to_idx_path):
            try:
                import json
                with open(class_to_idx_path, 'r', encoding='utf-8') as f:
                    class_to_idx = json.load(f)
                # 转换为按索引排序的类别列表
                idx_to_class = {int(v): k for k, v in class_to_idx.items()}
                classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                print(f"从class_to_idx.json文件加载了 {len(classes)} 个类别")
                return classes
            except Exception as e:
                print(f"从class_to_idx.json文件加载类别失败: {e}")
        
        # 3. 尝试从class_mapping.json文件加载
        class_mapping_path = os.path.join(os.path.dirname(self.model_path), f"{os.path.basename(self.model_path).split('.')[0]}_class_mapping.json")
        if not os.path.exists(class_mapping_path):
            # 尝试默认的class_mapping文件
            class_mapping_path = os.path.join(os.path.dirname(self.model_path), "character_classifier_best_improved_class_mapping.json")
        
        if os.path.exists(class_mapping_path):
            try:
                import json
                with open(class_mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                if 'class_to_idx' in mapping:
                    class_to_idx = mapping['class_to_idx']
                    # 转换为按索引排序的类别列表
                    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
                    classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                    print(f"从class_mapping.json文件加载了 {len(classes)} 个类别")
                    return classes
            except Exception as e:
                print(f"从class_mapping.json文件加载类别失败: {e}")
        
        # 4. 如果从模型文件加载失败，回退到从目录加载
        if not os.path.exists(self.data_dir):
            # 尝试从train目录加载
            train_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'train'))
            if os.path.exists(train_data_dir):
                self.data_dir = train_data_dir
                print(f"使用train目录作为数据目录: {self.data_dir}")
            else:
                raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
            
        # 必须与训练时一致的排序逻辑
        # 过滤掉非目录文件，如 .DS_Store
        classes = [d for d in os.listdir(self.data_dir) 
                  if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')]
        classes.sort()
        print(f"从目录加载了 {len(classes)} 个类别")
        return classes

    def _load_model(self):
        """加载模型结构和权重"""
        print(f"加载模型: {self.model_path}")
        num_classes = len(self.classes)
        
        # 根据模型文件路径选择合适的模型结构
        model = None
        if 'mobilenet_v2' in self.model_path:
            # 使用MobileNetV2模型结构
            print("使用MobileNetV2模型结构")
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            # 默认使用EfficientNetB0模型结构
            print("使用EfficientNetB0模型结构")
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # 加载权重
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理可能的不同保存格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 修复键名不匹配问题
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # 处理不同模型的键名前缀
                if k.startswith('backbone.'):
                    name = k[9:] # 移除 'backbone.'
                elif k.startswith('module.'):
                    name = k[7:] # 移除 'module.'
                else:
                    name = k
                new_state_dict[name] = v
                
            # 尝试加载权重，允许非严格匹配（如果还有其他微小差异）
            model.load_state_dict(new_state_dict, strict=False)
            print("模型权重加载成功 (已处理键名不匹配)")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

        model = model.to(self.device)
        model.eval()
        
        # 应用优化
        if self.enable_optimizations:
            model = self._optimize_model(model)
        
        return model

    def _optimize_model(self, model):
        """优化模型以提高推理速度"""
        print("应用模型优化...")
        
        # 1. 启用FP16精度（如果支持）
        if torch.cuda.is_available():
            model = model.half()
            print("启用FP16精度")
        
        # 2. 启用CUDA图（如果支持）
        if torch.cuda.is_available():
            try:
                # 创建示例输入
                sample_input = torch.randn(1, 3, 224, 224, device=self.device)
                if torch.cuda.is_available():
                    sample_input = sample_input.half()
                
                # 预热模型
                with torch.no_grad():
                    for _ in range(3):
                        model(sample_input)
                
                print("模型优化完成")
            except Exception as e:
                print(f"CUDA图优化失败: {e}")
        
        return model

    def _get_transforms(self):
        """预处理转换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path, top_k=5):
        """预测图片角色"""
        start_time = time.time()
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 如果启用了FP16，转换输入
            if self.enable_optimizations and torch.cuda.is_available():
                image_tensor = image_tensor.half()
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
            # 获取Top-K结果
            # 确保 k 不超过类别总数
            k = min(top_k, len(self.classes))
            top_prob, top_idx = torch.topk(probabilities, k)
            
            results = []
            for i in range(k):
                idx = top_idx[0][i].item()
                prob = top_prob[0][i].item()
                # 确保索引不越界
                if idx < len(self.classes):
                    role_name = self.classes[idx]
                    results.append({
                        "role": role_name,
                        "similarity": prob
                    })
            
            if not results:
                return "Unknown", 0.0, []
                
            # 返回最佳结果和完整列表
            best_role = results[0]["role"]
            best_score = results[0]["similarity"]
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) % 100 == 0:
                avg_time = sum(self.inference_times) / len(self.inference_times)
                print(f"平均推理时间: {avg_time:.4f}秒")
            
            return best_role, best_score, results
            
        except Exception as e:
            print(f"推理失败: {e}")
            return None, 0.0, []

    def predict_batch(self, image_paths, batch_size=32, top_k=5):
        """批量预测多张图片"""
        if not image_paths:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # 分批处理
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = []
                valid_indices = []
                
                # 加载和预处理批量图像
                for j, img_path in enumerate(batch_paths):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image_tensor = self.transform(image)
                        batch_images.append(image_tensor)
                        valid_indices.append(j)
                    except Exception as e:
                        print(f"加载图像 {img_path} 失败: {e}")
                        results.append({"image_path": img_path, "error": str(e)})
                
                if not batch_images:
                    continue
                
                # 创建批量张量
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # 如果启用了FP16，转换输入
                if self.enable_optimizations and torch.cuda.is_available():
                    batch_tensor = batch_tensor.half()
                
                # 批量推理
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # 获取Top-K结果
                    k = min(top_k, len(self.classes))
                    top_probs, top_idxs = torch.topk(probabilities, k)
                
                # 处理批量结果
                for j, (img_path, probs, idxs) in enumerate(zip(
                    [batch_paths[idx] for idx in valid_indices], 
                    top_probs, 
                    top_idxs
                )):
                    img_results = []
                    for prob, idx in zip(probs, idxs):
                        idx = idx.item()
                        prob = prob.item()
                        if idx < len(self.classes):
                            role_name = self.classes[idx]
                            img_results.append({
                                "role": role_name,
                                "similarity": prob
                            })
                    
                    if img_results:
                        best_role = img_results[0]["role"]
                        best_score = img_results[0]["similarity"]
                        results.append({
                            "image_path": img_path,
                            "best_role": best_role,
                            "best_score": best_score,
                            "all_results": img_results
                        })
                    else:
                        results.append({
                            "image_path": img_path,
                            "best_role": "Unknown",
                            "best_score": 0.0,
                            "all_results": []
                        })
            
            # 计算批量处理时间
            total_time = time.time() - start_time
            avg_time_per_image = total_time / len(image_paths)
            print(f"批量处理完成: {len(image_paths)}张图像，平均每张 {avg_time_per_image:.4f}秒")
            
            return results
            
        except Exception as e:
            print(f"批量推理失败: {e}")
            return []

    def predict_parallel(self, image_paths, num_workers=4, top_k=5):
        """并行预测多张图片"""
        if not image_paths:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有预测任务
                future_to_path = {executor.submit(self.predict, path, top_k): path for path in image_paths}
                
                # 收集结果
                for future in future_to_path:
                    img_path = future_to_path[future]
                    try:
                        best_role, best_score, all_results = future.result()
                        results.append({
                            "image_path": img_path,
                            "best_role": best_role,
                            "best_score": best_score,
                            "all_results": all_results
                        })
                    except Exception as e:
                        print(f"处理图像 {img_path} 失败: {e}")
                        results.append({
                            "image_path": img_path,
                            "error": str(e)
                        })
            
            # 计算并行处理时间
            total_time = time.time() - start_time
            avg_time_per_image = total_time / len(image_paths)
            print(f"并行处理完成: {len(image_paths)}张图像，平均每张 {avg_time_per_image:.4f}秒")
            
            return results
            
        except Exception as e:
            print(f"并行推理失败: {e}")
            return []

    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return {
                "total_inferences": 0,
                "average_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0
            }
        
        return {
            "total_inferences": len(self.inference_times),
            "average_time": sum(self.inference_times) / len(self.inference_times),
            "min_time": min(self.inference_times),
            "max_time": max(self.inference_times)
        }

    def clear_stats(self):
        """清除性能统计信息"""
        self.inference_times = []
        print("性能统计信息已清除")

if __name__ == "__main__":
    # 测试代码
    try:
        infer = EfficientNetInference()
        print("模型初始化成功")
    except Exception as e:
        print(f"初始化失败: {e}")
