#!/usr/bin/env python3
"""
视频角色检测脚本
使用训练好的模型检测视频中的角色
"""
import os
import argparse
import logging
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video_character_detection')


class CharacterClassifier(nn.Module):
    """角色分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型
        from torchvision import models
        self.backbone = models.efficientnet_b0(pretrained=False)
        # 替换分类头
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


class VideoCharacterDetector:
    """视频角色检测器"""
    
    def __init__(self, model_path, device='mps'):
        """初始化检测器
        
        Args:
            model_path: 模型路径
            device: 运行设备
        """
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.num_classes = self.checkpoint.get('num_classes', 26)
        self.model = CharacterClassifier(num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # 类别映射
        if 'class_to_idx' in self.checkpoint:
            class_to_idx = self.checkpoint['class_to_idx']
            self.class_names = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])
            self.num_classes = len(self.class_names)
        else:
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"模型加载完成，支持 {len(self.class_names)} 个类别")
    
    def detect_frame(self, frame):
        """检测单帧中的角色
        
        Args:
            frame: 视频帧
            
        Returns:
            检测结果
        """
        # 转换为PIL图像
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 预处理
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            confidence = torch.softmax(output, 1)[0][predicted.item()].item()
        
        # 获取结果
        class_idx = predicted.item()
        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'class_{class_idx}'
        
        return {
            'class_name': class_name,
            'confidence': confidence,
            'class_idx': class_idx
        }
    
    def process_video(self, input_path, output_path=None, display=False, frame_skip=1):
        """处理视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            display: 是否显示
            frame_skip: 帧跳过
        """
        # 打开视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {input_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 初始化输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detected_frames = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 跳过帧
                if frame_count % frame_skip != 0:
                    if out:
                        out.write(frame)
                    continue
                
                # 检测角色
                result = self.detect_frame(frame)
                detected_frames += 1
                
                # 在帧上绘制结果
                if result['confidence'] > 0.5:
                    label = f"{result['class_name']}: {result['confidence']:.2f}"
                    cv2.putText(frame, label, (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示帧
                if display:
                    cv2.imshow('Video Character Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 写入输出视频
                if out:
                    out.write(frame)
                
                # 打印进度
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"处理进度: {progress:.1f}%, 已检测 {detected_frames} 帧")
                    
        finally:
            # 释放资源
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        logger.info(f"视频处理完成，总帧数: {frame_count}, 检测帧数: {detected_frames}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视频角色检测工具')
    
    parser.add_argument('--model_path', type=str, 
                        default='models/character_classifier_best_improved.pth', 
                        help='模型路径')
    parser.add_argument('--input_video', type=str, required=True, 
                        help='输入视频路径')
    parser.add_argument('--output_video', type=str, 
                        default=None, 
                        help='输出视频路径')
    parser.add_argument('--display', action='store_true', 
                        help='显示视频')
    parser.add_argument('--frame_skip', type=int, default=1, 
                        help='帧跳过')
    parser.add_argument('--device', type=str, default='mps', 
                        help='运行设备')
    
    args = parser.parse_args()
    
    logger.info('开始视频角色检测...')
    logger.info(f'模型路径: {args.model_path}')
    logger.info(f'输入视频: {args.input_video}')
    logger.info(f'输出视频: {args.output_video}')
    
    # 创建检测器
    detector = VideoCharacterDetector(args.model_path, args.device)
    
    # 处理视频
    detector.process_video(
        args.input_video,
        args.output_video,
        args.display,
        args.frame_skip
    )
    
    logger.info('视频角色检测完成！')


if __name__ == "__main__":
    main()