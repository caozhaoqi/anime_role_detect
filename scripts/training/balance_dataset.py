import os
import json
import logging
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('balance_dataset')

TARGET_COUNT = 60

AUGMENTATION_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

def load_image(img_path):
    try:
        image = Image.open(img_path).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"无法加载图片 {img_path}: {e}")
        return None

def save_tensor_as_image(tensor_img, output_path):
    try:
        tensor_img = tensor_img.cpu().detach()
        tensor_img = torch.clamp(tensor_img, 0, 1)
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(tensor_img)
        pil_img.save(output_path, quality=95)
        return True
    except Exception as e:
        logger.error(f"保存图片失败 {output_path}: {e}")
        return False

def generate_augmented_images(class_dir, target_count=TARGET_COUNT):
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    current_count = len(images)
    if current_count == 0:
        logger.warning(f"类别 {os.path.basename(class_dir)} 没有图片，跳过")
        return 0

    if current_count >= target_count:
        logger.info(f"类别 {os.path.basename(class_dir)} 已有 {current_count} 张图片，已达标")
        return 0

    images_to_generate = target_count - current_count
    logger.info(f"类别 {os.path.basename(class_dir)}: {current_count} -> {target_count} (需生成 {images_to_generate} 张)")

    augmented_count = 0
    image_paths = [os.path.join(class_dir, f) for f in images]

    for i in range(images_to_generate):
        src_img_path = image_paths[i % current_count]
        src_image = load_image(src_img_path)

        if src_image is None:
            continue

        try:
            tensor_img = AUGMENTATION_TRANSFORM(src_image)

            base_name = os.path.splitext(os.path.basename(src_img_path))[0]
            output_name = f"{base_name}_aug_{i:04d}.jpg"
            output_path = os.path.join(class_dir, output_name)

            if save_tensor_as_image(tensor_img, output_path):
                augmented_count += 1

        except Exception as e:
            logger.error(f"增强图片失败 {src_img_path}: {e}")
            continue

        if (i + 1) % 10 == 0:
            logger.info(f"  已生成 {i + 1}/{images_to_generate} 张")

    return augmented_count

def main():
    parser = argparse.ArgumentParser(description='平衡数据集 - 为样本少的类别生成增强图片')
    parser.add_argument('--data_dir', default='./data/role_images', help='数据目录')
    parser.add_argument('--target_count', type=int, default=60, help='每个类别的目标图片数量')
    parser.add_argument('--backup', action='store_true', help='是否备份原图')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("开始平衡数据集")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"目标数量: {args.target_count} 张/类别")

    if not os.path.exists(args.data_dir):
        logger.error(f"数据目录不存在: {args.data_dir}")
        return

    classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    logger.info(f"发现 {len(classes)} 个类别")

    class_stats = []
    total_original = 0
    total_augmented = 0

    for class_name in tqdm(classes, desc="处理类别"):
        class_dir = os.path.join(args.data_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        original_count = len(images)

        augmented = generate_augmented_images(class_dir, args.target_count)

        class_stats.append({
            'class': class_name,
            'original': original_count,
            'augmented': augmented,
            'final': original_count + augmented
        })

        total_original += original_count
        total_augmented += augmented

    logger.info("\n" + "=" * 60)
    logger.info("数据集平衡完成")
    logger.info("=" * 60)
    logger.info(f"总类别数: {len(classes)}")
    logger.info(f"原始图片总数: {total_original}")
    logger.info(f"新增增强图片: {total_augmented}")
    logger.info(f"最终图片总数: {total_original + total_augmented}")
    logger.info(f"平均每类图片: {(total_original + total_augmented) / len(classes):.1f}")

    stats_path = os.path.join(args.data_dir, 'augmentation_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(class_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息已保存: {stats_path}")

    logger.info("\n类别统计 (原始 -> 最终):")
    for stat in sorted(class_stats, key=lambda x: x['original']):
        logger.info(f"  {stat['class']}: {stat['original']:3d} -> {stat['final']:3d} (+{stat['augmented']})")

if __name__ == '__main__':
    main()
