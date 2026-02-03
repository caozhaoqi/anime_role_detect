#!/usr/bin/env python3
"""
视频检测分析脚本
分析模型在二次元动漫视频上的检测效果
"""
import os
import cv2
import torch
import torchvision.transforms as transforms
import time
import argparse
import logging
import numpy as np
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analyze_video_detection')


class CharacterClassifier(torch.nn.Module):
    """角色分类模型"""
    
    def __init__(self, num_classes):
        super(CharacterClassifier, self).__init__()
        # 使用EfficientNet-B0
        from torchvision.models import efficientnet_b0
        self.backbone = efficientnet_b0(pretrained=False)
        # 替换分类头
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = torch.nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class VideoDetectionAnalyzer:
    """视频检测分析器"""
    
    def __init__(self, model_path='models/character_classifier_best_improved.pth', device='mps'):
        """初始化分析器
        
        Args:
            model_path: 模型路径
            device: 设备
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = []
        self.num_classes = 0
        self.transform = None
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            # 加载模型
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 提取分类信息
            if 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                self.class_names = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])
                self.num_classes = len(self.class_names)
                logger.info(f"模型包含 {self.num_classes} 个角色类别")
            elif 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
                self.num_classes = len(self.class_names)
                logger.info(f"模型包含 {self.num_classes} 个角色类别")
            else:
                logger.error("模型中未找到分类信息")
                return
            
            # 初始化模型
            self.model = CharacterClassifier(self.num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 定义预处理
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info(f"模型加载成功: {self.model_path}")
            logger.info(f"使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"初始化模型时出错: {e}")
    
    def preprocess_frame(self, frame):
        """预处理帧
        
        Args:
            frame: 视频帧
            
        Returns:
            预处理后的张量
        """
        try:
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 应用变换
            frame_tensor = self.transform(frame_rgb)
            # 添加批次维度
            frame_tensor = frame_tensor.unsqueeze(0)
            return frame_tensor.to(self.device)
        except Exception as e:
            logger.error(f"预处理帧时出错: {e}")
            return None
    
    def predict_frame(self, frame_tensor):
        """预测帧
        
        Args:
            frame_tensor: 预处理后的帧张量
            
        Returns:
            (预测类别, 置信度)
        """
        try:
            with torch.no_grad():
                outputs = self.model(frame_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                class_idx = predicted.item()
                confidence = confidence.item()
                
                # 添加置信度阈值过滤
                if confidence < 0.6:
                    return "No Character", confidence
                
                if 0 <= class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                    return class_name, confidence
                else:
                    return "Unknown", confidence
                    
        except Exception as e:
            logger.error(f"预测帧时出错: {e}")
            return "Unknown", 0.0
    
    def analyze_video(self, video_path, output_video=None, frame_skip=1):
        """分析视频
        
        Args:
            video_path: 视频路径
            output_video: 输出视频路径
            frame_skip: 帧跳过
            
        Returns:
            分析结果
        """
        if not self.model:
            logger.error("模型未初始化")
            return None
        
        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return None
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"分析视频: {video_path}")
            logger.info(f"视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")
            
            # 初始化视频编写器
            out = None
            if output_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # 分析结果
            results = {
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'total_frames': total_frames,
                'analyzed_frames': 0,
                'detection_results': [],
                'detection_times': [],
                'detected_characters': defaultdict(int),
                'confidence_scores': [],
                'fps': fps,
                'resolution': f"{width}x{height}"
            }
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 跳过帧
                if frame_count % frame_skip != 0:
                    continue
                
                # 预处理
                preprocess_start = time.time()
                frame_tensor = self.preprocess_frame(frame)
                preprocess_time = time.time() - preprocess_start
                
                if frame_tensor is None:
                    continue
                
                # 预测
                predict_start = time.time()
                predicted_class, confidence = self.predict_frame(frame_tensor)
                predict_time = time.time() - predict_start
                
                total_process_time = preprocess_time + predict_time
                
                # 记录结果
                results['analyzed_frames'] += 1
                results['detection_times'].append(total_process_time)
                results['confidence_scores'].append(confidence)
                
                if predicted_class != "Unknown":
                    results['detected_characters'][predicted_class] += 1
                    
                # 记录详细结果
                results['detection_results'].append({
                    'frame': frame_count,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'process_time': total_process_time
                })
                
                # 绘制结果
                if out:
                    display_frame = frame.copy()
                    
                    # 获取英文角色名
                    def get_english_name(chinese_name):
                        # 角色中英文映射
                        name_map = {
                            "原神_温迪": "Venti",
                            "原神_胡桃": "Hu Tao",
                            "原神_可莉": "Klee",
                            "原神_魈": "Xiao",
                            "原神_甘雨": "Ganyu",
                            "原神_钟离": "Zhongli",
                            "原神_琴": "Jean",
                            "原神_迪卢克": "Diluc",
                            "原神_凯亚": "Kaeya",
                            "原神_安柏": "Amber",
                            "原神_丽莎": "Lisa",
                            "原神_芭芭拉": "Barbara",
                            "原神_雷泽": "Razor",
                            "原神_诺艾尔": "Noelle",
                            "原神_菲谢尔": "Fischl",
                            "原神_砂糖": "Sucrose",
                            "原神_莫娜": "Mona",
                            "原神_空": "Aether",
                            "原神_荧": "Lumine",
                            "原神_阿贝多": "Albedo",
                            "原神_罗莎莉亚": "Rosaria",
                            "原神_优菈": "Eula",
                            "原神_烟绯": "Yanfei",
                            "原神_迪奥娜": "Diona",
                            "原神_早柚": "Sayu",
                            "原神_荒泷一斗": "Arataki Itto",
                            "原神_五郎": "Gorou",
                            "原神_八重神子": "Yae Miko",
                            "原神_神里绫华": "Kamisato Ayaka",
                            "原神_枫原万叶": "Kaedehara Kazuha",
                            "我推的孩子_星野爱": "Hoshino Ai",
                            "明日方舟_阿米娅": "Amiya",
                            "明日方舟_能天使": "Exusiai",
                            "明日方舟_德克萨斯": "Texas",
                            "明日方舟_陈": "Ch'en",
                            "蔚蓝档案_星野": "Hoshino",
                            "蔚蓝档案_日奈": "Hina",
                            "蔚蓝档案_阿罗娜": "Arona",
                            "蔚蓝档案_白子": "Shiromi",
                            "蔚蓝档案_优花梨": "Yuuka",
                            "蔚蓝档案_宫子": "Miyako",
                            "幻塔_凛夜": "Lin Ye",
                            "鸣潮_守岸人": "Shou An Ren",
                            "鸣潮_椿": "Tsubaki",
                            "绝区零_安比": "Anby",
                            "绝区零_杰克": "Jack",
                            "No Character": "No Character"
                        }
                        return name_map.get(chinese_name, chinese_name)
                    
                    # 获取英文角色名
                    english_name = get_english_name(predicted_class)
                    
                    text_y = 100
                    
                    # 只有检测到有效角色时才绘制框选和详细标注
                    if predicted_class != "No Character" and predicted_class != "Unknown":
                        # 人脸检测和跟踪
                        def detect_face(frame):
                            """检测人脸
                            
                            Args:
                                frame: 视频帧
                                
                            Returns:
                                (x1, y1, x2, y2) 或 None
                            """
                            try:
                                # 转换为灰度图
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                
                                # 方法1: 使用Haar级联分类器检测人脸（适用于真实人脸）
                                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                                
                                if len(faces) > 0:
                                    # 选择最大的人脸
                                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                                    x, y, w, h = largest_face
                                    return x, y, x + w, y + h
                                
                                # 方法2: 针对二次元角色的检测（基于颜色和形状）
                                # 转换为HSV颜色空间，检测浅色区域（通常是面部）
                                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                                
                                # 定义二次元角色面部的颜色范围（浅色皮肤）
                                lower_skin = (0, 20, 100)
                                upper_skin = (30, 255, 255)
                                
                                # 创建掩码
                                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                                
                                # 形态学操作，去除噪声
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                                
                                # 查找轮廓
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                # 筛选合适大小的轮廓（可能是面部）
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if 1000 < area < 100000:  # 调整面积范围
                                        x, y, w, h = cv2.boundingRect(contour)
                                        # 检查宽高比（面部通常接近正方形）
                                        aspect_ratio = w / float(h)
                                        if 0.7 < aspect_ratio < 1.3:
                                            return x, y, x + w, y + h
                                
                                # 方法3: 检测可能的头部区域（基于大小和位置）
                                height, width = frame.shape[:2]
                                # 检查图像中上部区域（通常是头部位置）
                                upper_region = frame[0:int(height*0.6), :]
                                upper_gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
                                
                                # 检测边缘
                                edges = cv2.Canny(upper_gray, 50, 150)
                                
                                # 查找轮廓
                                upper_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                for contour in upper_contours:
                                    area = cv2.contourArea(contour)
                                    if 500 < area < 50000:
                                        x, y, w, h = cv2.boundingRect(contour)
                                        aspect_ratio = w / float(h)
                                        if 0.5 < aspect_ratio < 1.5:
                                            return x, y, x + w, y + h
                                
                                return None
                            except Exception as e:
                                logger.warning(f"人脸检测出错: {e}")
                                return None
                        
                        # 检测人脸
                        face_bbox = detect_face(display_frame)
                        
                        if face_bbox:
                            # 人脸跟踪
                            x1, y1, x2, y2 = face_bbox
                            # 扩大框的大小以包含头部
                            padding = int((x2 - x1) * 0.3)
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding * 2)
                            x2 = min(display_frame.shape[1], x2 + padding)
                            y2 = min(display_frame.shape[0], y2 + padding)
                            
                            # 绘制人脸框
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            # 在框上方绘制检测结果
                            text_y = max(30, y1 - 10)
                        else:
                            # 没有检测到人脸时，不绘制框选，只显示文字信息
                            # 这样可以避免在非角色区域（如logo、文字页面）绘制错误的检测框
                            text_y = 100
                            # 跳过绘制框选的步骤
                        
                        # 只绘制英文角色名（避免中文乱码问题）
                        english_text = f"{english_name}: {confidence:.2f}"
                        cv2.putText(display_frame, english_text, 
                                   (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 255, 0), 2)
                        
                        # 移除中文标注，避免编码问题
                        # 英文标注已经足够清晰，且可以避免乱码
                    else:
                        # 未检测到有效角色时，显示"无角色"
                        cv2.putText(display_frame, "No Character Detected", 
                                   (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (255, 0, 0), 2)
                    
                    cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                               (50, text_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (255, 255, 255), 2)
                    
                    # 写入帧
                    out.write(display_frame)
                
                # 每100帧记录一次
                if frame_count % 100 == 0:
                    logger.info(f"处理帧 {frame_count}/{total_frames}")
            
            # 计算统计信息
            total_time = time.time() - start_time
            
            if results['analyzed_frames'] > 0:
                avg_process_time = np.mean(results['detection_times'])
                avg_confidence = np.mean(results['confidence_scores'])
                detection_rate = len(results['detected_characters']) / self.num_classes if self.num_classes > 0 else 0
                
                results.update({
                    'total_time': total_time,
                    'average_process_time': avg_process_time,
                    'average_confidence': avg_confidence,
                    'detection_rate': detection_rate,
                    'fps_processed': results['analyzed_frames'] / total_time
                })
                
                logger.info(f"分析完成: {results['analyzed_frames']}帧, 耗时 {total_time:.2f}秒, {results['fps_processed']:.2f}fps")
                logger.info(f"平均处理时间: {avg_process_time:.4f}秒/帧")
                logger.info(f"平均置信度: {avg_confidence:.2f}")
                logger.info(f"检出角色数: {len(results['detected_characters'])}/{self.num_classes}")
                logger.info(f"检出率: {detection_rate:.2f}")
            
            # 释放资源
            cap.release()
            if out:
                out.release()
            
            return results
            
        except Exception as e:
            logger.error(f"分析视频时出错: {e}")
            return None
    
    def analyze_multiple_videos(self, video_paths, output_dir=None):
        """分析多个视频
        
        Args:
            video_paths: 视频路径列表
            output_dir: 输出目录
            
        Returns:
            分析结果列表
        """
        all_results = []
        
        for video_path in video_paths:
            if output_dir:
                video_name = os.path.basename(video_path)
                output_video = os.path.join(output_dir, f"analyzed_{video_name}")
            else:
                output_video = None
            
            result = self.analyze_video(video_path, output_video)
            if result:
                all_results.append(result)
        
        return all_results
    
    def generate_report(self, results, report_path=None):
        """生成报告
        
        Args:
            results: 分析结果列表
            report_path: 报告路径
        """
        if not results:
            logger.error("无分析结果")
            return
        
        # 生成报告
        report = f"# 二次元角色视频检测分析报告\n\n"
        report += f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 总体统计
        total_videos = len(results)
        total_frames = sum(r.get('total_frames', 0) for r in results)
        total_analyzed_frames = sum(r.get('analyzed_frames', 0) for r in results)
        total_time = sum(r.get('total_time', 0) for r in results)
        
        avg_fps = total_analyzed_frames / total_time if total_time > 0 else 0
        avg_confidence = np.mean([r.get('average_confidence', 0) for r in results if 'average_confidence' in r])
        
        report += f"## 总体统计\n\n"
        report += f"| 指标 | 值 |\n"
        report += f"|------|-----|\n"
        report += f"| 分析视频数 | {total_videos} |\n"
        report += f"| 总帧数 | {total_frames} |\n"
        report += f"| 分析帧数 | {total_analyzed_frames} |\n"
        report += f"| 总耗时 | {total_time:.2f}秒 |\n"
        report += f"| 平均处理速度 | {avg_fps:.2f}fps |\n"
        report += f"| 平均置信度 | {avg_confidence:.2f} |\n\n"
        
        # 详细结果
        report += f"## 详细结果\n\n"
        
        for i, result in enumerate(results, 1):
            report += f"### 视频 {i}: {result.get('video_name', '未知')}\n\n"
            report += f"| 指标 | 值 |\n"
            report += f"|------|-----|\n"
            report += f"| 视频路径 | {result.get('video_path', '未知')} |\n"
            report += f"| 分辨率 | {result.get('resolution', '未知')} |\n"
            report += f"| 帧率 | {result.get('fps', '未知'):.2f}fps |\n"
            report += f"| 总帧数 | {result.get('total_frames', 0)} |\n"
            report += f"| 分析帧数 | {result.get('analyzed_frames', 0)} |\n"
            report += f"| 总耗时 | {result.get('total_time', 0):.2f}秒 |\n"
            report += f"| 处理速度 | {result.get('fps_processed', 0):.2f}fps |\n"
            report += f"| 平均置信度 | {result.get('average_confidence', 0):.2f} |\n"
            report += f"| 检出角色数 | {len(result.get('detected_characters', {}))} |\n"
            report += f"| 检出率 | {result.get('detection_rate', 0):.2f} |\n\n"
            
            # 检出角色
            detected_chars = result.get('detected_characters', {})
            if detected_chars:
                report += f"#### 检出角色\n\n"
                report += f"| 角色 | 检出次数 |\n"
                report += f"|------|----------|\n"
                
                for char, count in sorted(detected_chars.items(), key=lambda x: x[1], reverse=True):
                    report += f"| {char} | {count} |\n"
                report += "\n"
        
        # 保存报告
        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告保存到: {report_path}")
        
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视频检测分析工具')
    
    parser.add_argument('--videos', nargs='+',
                        default=['data/videos/anime_character_real_1.mp4',
                                 'data/videos/anime_character_real_2.mp4',
                                 'data/videos/anime_character_real_3.mp4'],
                        help='要分析的视频路径')
    
    parser.add_argument('--model', type=str,
                        default='models/character_classifier_best_improved.pth',
                        help='模型路径')
    
    parser.add_argument('--output_dir', type=str,
                        default='data/videos/analyzed',
                        help='分析结果输出目录')
    
    parser.add_argument('--report', type=str,
                        default='reports/video_detection_analysis.md',
                        help='分析报告路径')
    
    parser.add_argument('--device', type=str,
                        default='mps',
                        choices=['mps', 'cpu', 'cuda'],
                        help='运行设备')
    
    args = parser.parse_args()
    
    logger.info('开始视频检测分析...')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    
    # 初始化分析器
    analyzer = VideoDetectionAnalyzer(args.model, args.device)
    
    # 分析视频
    results = analyzer.analyze_multiple_videos(args.videos, args.output_dir)
    
    # 生成报告
    if results:
        report = analyzer.generate_report(results, args.report)
        print(report)
    
    logger.info('视频检测分析完成！')


if __name__ == "__main__":
    main()