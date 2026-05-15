#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二次元角色识别模型训练
支持飞书推送训练进展
"""
import os
import sys
import time
import json
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 配置路径
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset'
MODEL_SAVE_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/models'
LOG_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/logs'

# 训练配置
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)

# 加载飞书配置
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notification_config.json')
notification_available = False
notification_config = {}

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        notification_config = json.load(f)
    notification_available = True
    print(f"已加载通知配置: {config_path}")
else:
    print(f"未找到通知配置文件: {config_path}")

# 创建必要目录
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def send_feishu_notification(message, title="模型训练通知"):
    """发送飞书通知"""
    if not notification_available:
        return False
    
    try:
        import requests
        
        # 获取飞书token
        token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        token_data = {
            "app_id": notification_config['feishu']['app_id'],
            "app_secret": notification_config['feishu']['app_secret']
        }
        
        response = requests.post(token_url, json=token_data)
        if response.status_code != 200:
            logger.warning("获取飞书token失败")
            return False
        
        token = response.json().get('tenant_access_token')
        
        # 发送消息
        send_url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        msg_data = {
            "receive_id": notification_config['feishu']['receive_id'],
            "receive_id_type": notification_config['feishu']['receive_id_type'],
            "content": json.dumps({
                "text": f"**{title}**\n\n{message}"
            }),
            "msg_type": "text"
        }
        
        response = requests.post(send_url, headers=headers, json=msg_data)
        if response.status_code == 200:
            logger.info("飞书通知发送成功")
            return True
        else:
            logger.warning(f"飞书通知发送失败: {response.text}")
            return False
            
    except Exception as e:
        logger.warning(f"发送飞书通知异常: {e}")
        return False


def count_classes_and_images():
    """统计数据集中的类别数和图片数"""
    classes = []
    total_images = 0
    
    for item in os.listdir(DATASET_PATH):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            classes.append(item)
            images = [f for f in os.listdir(item_path) if f.lower().endswith('.jpg')]
            total_images += len(images)
    
    return sorted(classes), total_images


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🎬 二次元角色识别模型训练")
    logger.info("=" * 60)
    
    try:
        # 发送开始通知
        send_feishu_notification(
            f"数据集: {DATASET_PATH}\n配置: {EPOCHS} 轮, 批次 {BATCH_SIZE}, 学习率 {LEARNING_RATE}\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "🚀 模型训练任务开始"
        )
        
        # 准备数据
        logger.info("📦 准备训练数据...")
        
        if not os.path.exists(DATASET_PATH):
            logger.error(f"数据集路径不存在: {DATASET_PATH}")
            send_feishu_notification(f"数据集路径不存在: {DATASET_PATH}", "❌ 训练异常")
            return
        
        classes, total_images = count_classes_and_images()
        num_classes = len(classes)
        
        logger.info(f"  ✅ 发现 {num_classes} 个角色类别")
        logger.info(f"  ✅ 总计 {total_images} 张图片")
        
        # 构建模型
        logger.info("🏗️ 构建模型...")
        
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.models import Model
        
        base_model = MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        inputs = base_model.input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("  ✅ 模型构建完成")
        
        # 训练模型
        logger.info("🚀 开始训练...")
        
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='sparse',
            subset='training',
            shuffle=True
        )
        
        val_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='sparse',
            subset='validation',
            shuffle=True
        )
        
        best_val_acc = 0.0
        
        for epoch in range(1, EPOCHS + 1):
            logger.info(f"\n📊 Epoch {epoch}/{EPOCHS}")
            
            history = model.fit(
                train_generator,
                epochs=1,
                validation_data=val_generator,
                verbose=1
            )
            
            train_acc = history.history['accuracy'][0]
            train_loss = history.history['loss'][0]
            val_acc = history.history['val_accuracy'][0]
            val_loss = history.history['val_loss'][0]
            
            logger.info(f"  训练准确率: {train_acc:.4f}, 训练损失: {train_loss:.4f}")
            logger.info(f"  验证准确率: {val_acc:.4f}, 验证损失: {val_loss:.4f}")
            
            # 每5轮发送一次进度通知
            if epoch % 5 == 0:
                send_feishu_notification(
                    f"轮次: Epoch {epoch}/{EPOCHS}\n训练准确率: {train_acc:.4f}\n验证准确率: {val_acc:.4f}\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"🔄 训练进度更新"
                )
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(MODEL_SAVE_PATH, f'best_model_epoch_{epoch}.h5')
                model.save(model_path)
                logger.info(f"  🎯 保存最佳模型: {model_path}")
        
        logger.info("\n🎉 训练完成！")
        
        # 保存最终模型
        final_model_path = os.path.join(MODEL_SAVE_PATH, f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        model.save(final_model_path)
        logger.info(f"✅ 保存最终模型: {final_model_path}")
        
        # 发送完成通知
        summary = f"""训练配置:
- 数据集: {len(classes)} 个角色, {total_images} 张图片
- 轮次: {EPOCHS}
- 批次: {BATCH_SIZE}

训练结果:
- ✅ 最佳验证准确率: {best_val_acc:.4f}
- 📁 模型保存路径: {MODEL_SAVE_PATH}

时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
        
        send_feishu_notification(summary, "🎉 训练任务完成")
        
        logger.info("=" * 60)
        logger.info("🎬 训练任务结束")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        send_feishu_notification(f"训练失败: {str(e)}", "❌ 训练异常")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()