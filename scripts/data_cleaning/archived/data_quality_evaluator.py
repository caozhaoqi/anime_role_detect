import os
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

class DataQualityEvaluator:
    def __init__(self, data_dir, output_dir=None):
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(os.path.dirname(data_dir), 'quality_filtered')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_quality_score(self, image_path):
        """计算图片质量分数"""
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. 清晰度评分 (拉普拉斯方差)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # 2. 对比度评分
            contrast = gray.std()
            
            # 3. 亮度评分 (避免过暗或过亮)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # 4. 模糊检测
            blur_score = min(sharpness / 100.0, 1.0)
            
            # 综合评分
            score = (blur_score * 0.4 + contrast / 100.0 * 0.3 + brightness_score * 0.3)
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 0.0
    
    def evaluate_dataset(self, threshold=0.3):
        """评估整个数据集的质量"""
        results = defaultdict(list)
        total_images = 0
        kept_images = 0
        removed_images = 0
        
        for char in os.listdir(self.data_dir):
            char_path = os.path.join(self.data_dir, char)
            if not os.path.isdir(char_path):
                continue
            
            char_output_dir = os.path.join(self.output_dir, char)
            os.makedirs(char_output_dir, exist_ok=True)
            
            for img_file in os.listdir(char_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(char_path, img_file)
                score = self.calculate_quality_score(img_path)
                
                if score >= threshold:
                    # 保留高质量图片
                    dest_path = os.path.join(char_output_dir, img_file)
                    os.symlink(img_path, dest_path)
                    kept_images += 1
                else:
                    removed_images += 1
                
                results[char].append({
                    'file': img_file,
                    'score': score
                })
                
                total_images += 1
        
        # 生成报告
        self._generate_report(results, total_images, kept_images, removed_images)
        return results
    
    def _generate_report(self, results, total, kept, removed):
        """生成质量评估报告"""
        report_path = os.path.join(self.output_dir, 'quality_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("数据质量评估报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"总图片数: {total}\n")
            f.write(f"保留图片数: {kept} ({kept/total*100:.1f}%)\n")
            f.write(f"移除图片数: {removed} ({removed/total*100:.1f}%)\n")
            f.write("\n各角色统计:\n")
            f.write("-" * 60 + "\n")
            
            for char, items in sorted(results.items(), key=lambda x: len(x[1]), reverse=True):
                avg_score = sum(item['score'] for item in items) / len(items)
                f.write(f"{char:20} 总数:{len(items):3} 平均分:{avg_score:.3f}\n")
        
        print(f"✅ 质量评估完成！报告已保存到: {report_path}")
        print(f"📊 保留 {kept} 张图片 ({kept/total*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据质量评估与筛选')
    parser.add_argument('--data_dir', type=str, required=True, help='输入数据目录')
    parser.add_argument('--threshold', type=float, default=0.3, help='质量阈值')
    args = parser.parse_args()
    
    evaluator = DataQualityEvaluator(args.data_dir)
    evaluator.evaluate_dataset(threshold=args.threshold)