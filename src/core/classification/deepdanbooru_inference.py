import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

class DeepDanbooruInference:
    _instance = None
    
    def __new__(cls, enable_optimizations=True):
        if cls._instance is None:
            cls._instance = super(DeepDanbooruInference, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, enable_optimizations=True):
        if self.initialized:
            return
            
        self.enable_optimizations = enable_optimizations
        self.api_url = "https://hysts-deepdanbooru.hf.space/api/predict"
        self.initialized = True
        
        # 性能统计
        self.inference_times = []

    def predict(self, image_path, top_k=10, threshold=0.5):
        """使用DeepDanbooru模型预测图片标签"""
        start_time = time.time()
        
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(image_path)
            if file_size > 10 * 1024 * 1024:  # 10MB限制
                raise ValueError("图像文件过大，最大支持10MB")
            
            # 准备请求数据
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f)}
                data = {
                    'top_k': top_k,
                    'threshold': threshold
                }
                
                # 发送请求
                response = requests.post(self.api_url, files=files, data=data, timeout=30)
                
                # 检查响应状态
                response.raise_for_status()
                
                # 解析响应
                result = response.json()
                
                # 提取标签和置信度
                tags = []
                if 'data' in result and isinstance(result['data'], list):
                    for item in result['data']:
                        if isinstance(item, dict) and 'tag' in item and 'score' in item:
                            tags.append({
                                "tag": item['tag'],
                                "score": item['score']
                            })
                
                # 按置信度排序
                tags.sort(key=lambda x: x['score'], reverse=True)
                
                # 记录推理时间
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                if len(self.inference_times) % 100 == 0:
                    avg_time = sum(self.inference_times) / len(self.inference_times)
                    print(f"DeepDanbooru平均推理时间: {avg_time:.4f}秒")
                
                return tags
                
        except Exception as e:
            print(f"DeepDanbooru推理失败: {e}")
            return []

    def predict_batch(self, image_paths, batch_size=4, top_k=10, threshold=0.5):
        """批量预测多张图片"""
        if not image_paths:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # 分批处理
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                
                # 对批次中的每张图像进行预测
                for img_path in batch_paths:
                    try:
                        tags = self.predict(img_path, top_k, threshold)
                        results.append({
                            "image_path": img_path,
                            "tags": tags
                        })
                    except Exception as e:
                        print(f"处理图像 {img_path} 失败: {e}")
                        results.append({
                            "image_path": img_path,
                            "error": str(e)
                        })
            
            # 计算批量处理时间
            total_time = time.time() - start_time
            avg_time_per_image = total_time / len(image_paths)
            print(f"DeepDanbooru批量处理完成: {len(image_paths)}张图像，平均每张 {avg_time_per_image:.4f}秒")
            
            return results
            
        except Exception as e:
            print(f"DeepDanbooru批量推理失败: {e}")
            return []

    def predict_parallel(self, image_paths, num_workers=4, top_k=10, threshold=0.5):
        """并行预测多张图片"""
        if not image_paths:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有预测任务
                future_to_path = {executor.submit(self.predict, path, top_k, threshold): path for path in image_paths}
                
                # 收集结果
                for future in future_to_path:
                    img_path = future_to_path[future]
                    try:
                        tags = future.result()
                        results.append({
                            "image_path": img_path,
                            "tags": tags
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
            print(f"DeepDanbooru并行处理完成: {len(image_paths)}张图像，平均每张 {avg_time_per_image:.4f}秒")
            
            return results
            
        except Exception as e:
            print(f"DeepDanbooru并行推理失败: {e}")
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
        print("DeepDanbooru性能统计信息已清除")

if __name__ == "__main__":
    # 测试代码
    try:
        infer = DeepDanbooruInference()
        print("DeepDanbooru推理器初始化成功")
        
        # 测试单张图像预测
        test_image = "test.jpg"
        if os.path.exists(test_image):
            tags = infer.predict(test_image)
            print(f"测试图像标签: {tags[:5]}")
        else:
            print(f"测试图像不存在: {test_image}")
            
    except Exception as e:
        print(f"初始化失败: {e}")
