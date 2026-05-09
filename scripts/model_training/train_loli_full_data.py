import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
import json
import logging
import time
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
PATIENCE = 15
EARLY_STOP_THRESHOLD = 0.001

def train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if batch_idx % 20 == 0 and phase == 'train':
                    logger.info(f'  Batch {batch_idx}/{len(dataloaders[phase])}, Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            logger.info(f'  {phase} Loss: {epoch_loss:.4f}, {phase} Acc: {epoch_acc:.4%}')
            
            if phase == 'val':
                if epoch_acc > best_acc + EARLY_STOP_THRESHOLD:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    logger.info(f'  保存最佳模型，准确率: {best_acc:.4%}')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f'  验证准确率连续 {patience} 轮未提升，提前停止训练')
                        model.load_state_dict(best_model_wts)
                        return model, best_acc
    
    time_elapsed = time.time() - since
    logger.info(f'训练完成，耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'最佳验证准确率: {best_acc:.4%}')
    
    model.load_state_dict(best_model_wts)
    return model, best_acc

def main():
    logger.info("="*60)
    logger.info("使用完整数据集训练角色分类模型")
    logger.info("="*60)
    
    data_dir = './data/organized_images'
    model_dir = './models/efficientnet_b0_loli_full_data'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"使用设备: {DEVICE}")
    logger.info(f"加载数据: {data_dir}")
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = ImageFolder(data_dir, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"数据集: {len(full_dataset)} 样本, {len(full_dataset.classes)} 类别")
    logger.info(f"训练集: {train_size}")
    logger.info(f"验证集: {val_size}")
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    
    class_to_idx = full_dataset.class_to_idx
    with open(Path(model_dir) / 'class_to_idx.json', 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    
    num_classes = len(full_dataset.classes)
    logger.info(f"\n创建模型: efficientnet_b0, 类别数: {num_classes}")
    
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"\n开始训练: efficientnet_b0")
    model, best_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    torch.save(model.state_dict(), Path(model_dir) / 'model_best.pth')
    torch.save(model, Path(model_dir) / 'model_full.pth')
    
    logger.info(f"模型已保存: {Path(model_dir) / 'model_best.pth'}")
    logger.info(f"efficientnet_b0 训练完成，准确率: {best_acc:.4%}")
    
    results = {
        'model': 'efficientnet_b0',
        'accuracy': best_acc.item(),
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_classes': num_classes,
        'train_samples': train_size,
        'val_samples': val_size
    }
    
    with open(Path(model_dir) / 'training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练结果已保存: {Path(model_dir) / 'training_results.json'}")
    logger.info("\n" + "="*60)
    logger.info("训练完成")
    logger.info("="*60)

if __name__ == '__main__':
    main()