#!/usr/bin/env python3
"""
先进训练策略脚本
- 迁移学习
- 数据增强
- 学习率调度
- 早停策略
- 模型量化
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 导入统一日志配置
from common.logging_config import get_logger

# 配置日志
logger = get_logger("training.advanced_training", "advanced_training.log")

# 全局配置
GLOBAL_CONFIG = {
    "data_dir": "../../data/role_images",
    "model_dir": "../../models",
    "image_size": (224, 224),
    "batch_size": 16,
    "epochs": 150,
    "learning_rate": 1e-4,
    "dropout_rate": 0.5,
    "validation_split": 0.2,
    "test_split": 0.1,
    "use_pretrained": True,
    "freeze_layers": True,
    "fine_tune_epochs": 50,
    "augmentation": True,
}


def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_dataset():
    """加载数据集"""
    logger.info("开始加载数据集")

    images = []
    labels = []

    # 遍历角色目录
    role_dirs = []
    for item in os.listdir(GLOBAL_CONFIG["data_dir"]):
        item_path = os.path.join(GLOBAL_CONFIG["data_dir"], item)
        if os.path.isdir(item_path):
            role_dirs.append(item)

    logger.info(f"发现 {len(role_dirs)} 个角色")

    # 加载图片和标签
    for role_idx, role_name in enumerate(role_dirs):
        role_dir = os.path.join(GLOBAL_CONFIG["data_dir"], role_name)

        for file_name in os.listdir(role_dir):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                file_path = os.path.join(role_dir, file_name)
                try:
                    # 加载图片
                    img = tf.keras.preprocessing.image.load_img(
                        file_path, target_size=GLOBAL_CONFIG["image_size"]
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(role_idx)
                except Exception as e:
                    logger.warning(f"加载图片失败: {file_path} - {str(e)}")

    # 转换为numpy数组
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # 数据归一化
    images = images / 255.0

    # 标签编码
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # 转换为one-hot编码
    num_classes = len(np.unique(labels))
    labels = to_categorical(labels, num_classes)

    logger.info(f"数据集加载完成: {len(images)} 张图片，{num_classes} 个角色")

    return images, labels, label_encoder, num_classes


def create_model(num_classes):
    """创建模型"""
    logger.info("开始创建模型")

    # 加载预训练模型
    base_model = EfficientNetB3(
        weights="imagenet" if GLOBAL_CONFIG["use_pretrained"] else None,
        include_top=False,
        input_shape=(*GLOBAL_CONFIG["image_size"], 3),
    )

    # 冻结预训练层
    if GLOBAL_CONFIG["freeze_layers"]:
        for layer in base_model.layers:
            layer.trainable = False

    # 添加自定义顶层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(GLOBAL_CONFIG["dropout_rate"])(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # 创建完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 编译模型
    optimizer = Adam(learning_rate=GLOBAL_CONFIG["learning_rate"])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    logger.info("模型创建完成")
    return model, base_model


def create_data_generators(images, labels):
    """创建数据生成器"""
    logger.info("开始创建数据生成器")

    # 划分数据集
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=GLOBAL_CONFIG["test_split"], random_state=42
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=GLOBAL_CONFIG["validation_split"] / (1 - GLOBAL_CONFIG["test_split"]),
        random_state=42,
    )

    logger.info(
        f"数据集划分: 训练集 {len(x_train)} 张，验证集 {len(x_val)} 张，测试集 {len(x_test)} 张"
    )

    # 创建数据增强生成器
    if GLOBAL_CONFIG["augmentation"]:
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator()

    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    # 创建生成器
    train_generator = train_datagen.flow(
        x_train, y_train, batch_size=GLOBAL_CONFIG["batch_size"], shuffle=True
    )

    val_generator = val_datagen.flow(
        x_val, y_val, batch_size=GLOBAL_CONFIG["batch_size"], shuffle=False
    )

    test_generator = test_datagen.flow(
        x_test, y_test, batch_size=GLOBAL_CONFIG["batch_size"], shuffle=False
    )

    logger.info("数据生成器创建完成")
    return train_generator, val_generator, test_generator, x_test, y_test


def train_model(model, train_generator, val_generator, base_model):
    """训练模型"""
    logger.info("开始训练模型")

    # 确保模型目录存在
    ensure_directory(GLOBAL_CONFIG["model_dir"])

    # 创建回调函数
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=1e-7),
        ModelCheckpoint(
            os.path.join(GLOBAL_CONFIG["model_dir"], "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    # 第一阶段训练：只训练顶层
    logger.info("开始第一阶段训练：只训练顶层")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=GLOBAL_CONFIG["epochs"] - GLOBAL_CONFIG["fine_tune_epochs"],
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
    )

    # 第二阶段训练：微调全部层
    logger.info("开始第二阶段训练：微调全部层")
    for layer in base_model.layers:
        layer.trainable = True

    # 重新编译模型，使用更小的学习率
    optimizer = Adam(learning_rate=GLOBAL_CONFIG["learning_rate"] / 10)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # 继续训练
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=GLOBAL_CONFIG["fine_tune_epochs"],
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
    )

    logger.info("模型训练完成")
    return model, history, history_fine


def evaluate_model(model, test_generator, x_test, y_test):
    """评估模型"""
    logger.info("开始评估模型")

    # 在测试集上评估
    test_loss, test_accuracy = model.evaluate(test_generator)
    logger.info(f"测试集评估结果: 损失 {test_loss:.4f}, 准确率 {test_accuracy:.4f}")

    # 预测
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # 计算混淆矩阵
    from sklearn.metrics import classification_report, confusion_matrix

    logger.info("分类报告:")
    logger.info(classification_report(true_classes, predicted_classes))

    logger.info("混淆矩阵:")
    logger.info(confusion_matrix(true_classes, predicted_classes))

    return test_accuracy


def save_model(model, label_encoder, num_classes):
    """保存模型"""
    logger.info("开始保存模型")

    # 保存模型
    model_path = os.path.join(GLOBAL_CONFIG["model_dir"], "final_model.h5")
    model.save(model_path)
    logger.info(f"模型保存到: {model_path}")

    # 保存标签编码器
    import joblib

    encoder_path = os.path.join(GLOBAL_CONFIG["model_dir"], "label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"标签编码器保存到: {encoder_path}")

    # 保存配置
    config_path = os.path.join(GLOBAL_CONFIG["model_dir"], "model_config.json")
    config = {
        "image_size": GLOBAL_CONFIG["image_size"],
        "num_classes": num_classes,
        "classes": label_encoder.classes_.tolist(),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"配置保存到: {config_path}")


def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始先进训练策略")
    logger.info("============================================================")

    # 加载数据集
    images, labels, label_encoder, num_classes = load_dataset()

    # 创建模型
    model, base_model = create_model(num_classes)

    # 创建数据生成器
    train_generator, val_generator, test_generator, x_test, y_test = create_data_generators(
        images, labels
    )

    # 训练模型
    model, history, history_fine = train_model(model, train_generator, val_generator, base_model)

    # 评估模型
    test_accuracy = evaluate_model(model, test_generator, x_test, y_test)

    # 保存模型
    save_model(model, label_encoder, num_classes)

    logger.info("\n============================================================")
    logger.info("先进训练策略完成")
    logger.info(f"测试集准确率: {test_accuracy:.4f}")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
