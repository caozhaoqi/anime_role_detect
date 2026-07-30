from PIL import Image
import numpy as np
import os
import gc

# 启用MPS和CUDA支持
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_HIGH_WATERMARK_RATIO"] = "0.5"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"

# 延迟导入torch（避免启动时锁竞争）
torch = None

# 延迟导入transformers模块
CLIPProcessor = None
CLIPModel = None

# 使用全局日志系统
from src.core.logging import get_enhanced_logger as get_logger, log_system, log_error
from src.core.config.device_manager import DeviceManager

logger = get_logger("feature_extraction")

# 导入跨平台诊断工具
try:
    from src.utils.diagnostics import CrossPlatformDiagnostics

    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    logger.warning("跨平台诊断工具不可用")
    DIAGNOSTICS_AVAILABLE = False

# 默认 EfficientNet 模型路径
EFFICIENTNET_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "models", "efficientnet_b3"
)

# 动态导入函数
def import_torch_modules():
    global torch, CLIPProcessor, CLIPModel, gc
    # 先导入torch
    import torch
    # 导入transformers模块
    from transformers import CLIPProcessor, CLIPModel
    import gc


class FeatureExtraction:
    # 全局模型实例缓存
    _model_instance = None
    _processor_instance = None
    _device = None
    _quantized = False
    _model_name = None
    _coreml_extractor = None
    _use_fp16 = True
    # EfficientNet 模型缓存（跨实例共享）
    _efficientnet_model = None
    _efficientnet_projection = None
    _efficientnet_transform = None

    def __init__(self, model_name="openai/clip-vit-base-patch32", quantize=True, use_coreml=False, use_fp16=True, use_clip=False, use_efficientnet=None):
        """初始化特征提取模块

        Args:
            model_name: 模型名称
            quantize: 是否使用量化模型
            use_coreml: 是否使用Core ML模式
            use_fp16: 是否使用FP16半精度推理
            use_clip: 是否使用CLIP模型（默认False，使用简单特征提取）
            use_efficientnet: 是否使用EfficientNet模型（默认自动检测）
        """
        # 禁用Core ML模式，避免锁竞争问题
        use_coreml = False

        # 延迟加载模型，避免初始化时的锁竞争
        self.model_name = model_name
        self.quantize = quantize
        self.coreml_mode = False
        self._use_fp16 = use_fp16 and not use_coreml
        self._use_clip = use_clip

        # 自动检测是否使用 EfficientNet
        if use_efficientnet is None:
            # macOS 上 CLIP 不可用，自动使用 EfficientNet
            import platform
            use_efficientnet = platform.system() == "Darwin"
        self._use_efficientnet = use_efficientnet

        # 初始化模型实例为None
        self.model = None
        self.processor = None
        self.efficientnet_model = None
        self.efficientnet_projection = None
        self.efficientnet_transform = None

        # 自动选择最佳设备（仅在需要时导入torch）
        self.device = self._select_device()

        # 加载模型
        if self._use_efficientnet:
            self._load_efficientnet_model()
        elif self._use_clip:
            self._load_model()

        # 确定模式描述
        if self._use_efficientnet and self.efficientnet_model is not None:
            mode = "EfficientNet"
        elif self.model is not None and self.processor is not None:
            mode = "CLIP"
        else:
            mode = "Simple"

        logger.info(f"特征提取模块使用设备: {self.device}")
        logger.info(f"特征提取模块初始化完成，模式: {mode}, FP16: {self._use_fp16}")
    
    def _select_device(self):
        """选择最佳计算设备（委托给 DeviceManager，保持 CUDA→MPS→CPU 检测顺序）"""
        device = DeviceManager.get_device()
        if device == "cpu":
            # CPU上不建议使用FP16
            self._use_fp16 = False
        return device

    def _load_model(self):
        """延迟加载CLIP模型"""
        try:
            # 导入transformers模块
            import_torch_modules()

            global CLIPProcessor, CLIPModel

            # 加载处理器和模型
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)

            # 将模型移动到指定设备
            self.model = self.model.to(self.device)

            # 启用FP16半精度推理
            if self._use_fp16 and (self.device == "cuda" or self.device == "mps"):
                self.model = self.model.half()
                logger.info("已启用FP16半精度推理")

            # 设置模型为评估模式
            self.model.eval()

            # 模型预热
            self._warmup_model()

            logger.info(f"成功加载CLIP模型: {self.model_name}")

        except Exception as e:
            logger.warning(f"加载CLIP模型失败，将使用简单特征提取方法: {e}")
            self.processor = None
            self.model = None

    def _load_efficientnet_model(self):
        """加载EfficientNet-B3模型作为特征提取骨干"""
        import torch
        import torchvision

        try:
            # 使用类级别缓存
            if FeatureExtraction._efficientnet_model is not None:
                self.efficientnet_model = FeatureExtraction._efficientnet_model
                self.efficientnet_projection = FeatureExtraction._efficientnet_projection
                self.efficientnet_transform = FeatureExtraction._efficientnet_transform
                logger.info("使用缓存的EfficientNet模型")
                return

            # 查找模型文件
            model_full_path = os.path.join(EFFICIENTNET_MODEL_DIR, "model_best.pth")
            if not os.path.exists(model_full_path):
                logger.warning(f"EfficientNet模型文件不存在: {model_full_path}，回退到简单方法")
                return

            logger.info(f"加载EfficientNet-B3模型: {model_full_path}")

            # 允许加载 EfficientNet 类（兼容不同PyTorch版本）
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(
                    [torchvision.models.efficientnet.EfficientNet]
                )

            # 加载模型（兼容不同PyTorch版本和不同保存格式）
            # 先加载到CPU，再移动到目标设备（避免MPS上的加载问题）
            try:
                loaded = torch.load(model_full_path, map_location='cpu', weights_only=False)
            except TypeError:
                loaded = torch.load(model_full_path, map_location='cpu')
            
            # 如果加载的是state_dict（字典），则创建模型实例并加载权重
            # 支持多种保存格式：完整模型、纯state_dict、训练检查点（包含model_state_dict）
            if isinstance(loaded, dict):
                if 'model_state_dict' in loaded:
                    state_dict = loaded['model_state_dict']
                elif any(k.startswith('features.') or k.startswith('classifier.') for k in loaded.keys()):
                    state_dict = loaded
                else:
                    logger.warning("无法识别模型文件格式，回退到简单方法")
                    return
                
                model = torchvision.models.efficientnet_b3(weights=None)
                
                # 检查state_dict中的分类器层，判断是否为自定义模型
                classifier_keys = [k for k in state_dict.keys() if k.startswith('classifier.')]
                features_keys = [k for k in state_dict.keys() if k.startswith('features.')]
                
                if classifier_keys and features_keys:
                    # 分析分类器结构
                    logger.info(f"检测到自定义分类器结构: {len(classifier_keys)} 个分类器参数")
                    
                    # 按索引分组
                    classifier_layers = {}
                    for key in classifier_keys:
                        parts = key.split('.')
                        if len(parts) >= 2:
                            try:
                                idx = int(parts[1])
                                if idx not in classifier_layers:
                                    classifier_layers[idx] = {}
                                param_name = '.'.join(parts[2:])
                                classifier_layers[idx][param_name] = state_dict[key]
                            except ValueError:
                                pass
                    
                    # 按索引排序
                    sorted_indices = sorted(classifier_layers.keys())
                    logger.info(f"分类器层索引: {sorted_indices}")
                    
                    # 重建分类器结构以匹配state_dict
                    new_classifier_modules = []
                    max_idx = max(sorted_indices)
                    
                    for idx in range(max_idx + 1):
                        if idx in classifier_layers:
                            layer_params = classifier_layers[idx]
                            if 'weight' in layer_params and 'bias' in layer_params:
                                # 检查是否为BatchNorm（有running_mean）
                                if 'running_mean' in layer_params:
                                    # BatchNorm层
                                    num_features = layer_params['weight'].shape[0]
                                    layer = torch.nn.BatchNorm1d(num_features)
                                    logger.info(f"  层 {idx}: BatchNorm1d({num_features})")
                                else:
                                    # Linear层
                                    weight_shape = layer_params['weight'].shape
                                    out_features, in_features = weight_shape[0], weight_shape[1] if len(weight_shape) > 1 else 768
                                    layer = torch.nn.Linear(in_features, out_features)
                                    logger.info(f"  层 {idx}: Linear({in_features}, {out_features})")
                                new_classifier_modules.append(layer)
                            else:
                                # 有参数但没有weight/bias，可能是其他类型，跳过
                                new_classifier_modules.append(torch.nn.Identity())
                                logger.info(f"  层 {idx}: Identity (未知类型)")
                        else:
                            # 缺失的层
                            if idx == 0:
                                # 标准EfficientNet在层0是Dropout
                                new_classifier_modules.append(torch.nn.Dropout(p=0.2, inplace=True))
                                logger.info(f"  层 {idx}: Dropout (自动补全)")
                            else:
                                # 其他缺失层通常是ReLU
                                new_classifier_modules.append(torch.nn.ReLU(inplace=True))
                                logger.info(f"  层 {idx}: ReLU (自动补全)")
                    
                    model.classifier = torch.nn.Sequential(*new_classifier_modules)
                    logger.info(f"重建分类器结构: {[type(m).__name__ for m in model.classifier]}")
                    
                    # 加载权重
                    model.load_state_dict(state_dict)
                    logger.info(f"成功加载 {len(state_dict)} 个权重参数")
                else:
                    # 标准模型，直接加载
                    model.load_state_dict(state_dict)
            else:
                model = loaded
            
            model.eval()
            
            # 如果是MPS设备，先在CPU上预热，然后再移动到MPS
            if self.device == "mps":
                logger.info("MPS设备：先在CPU上预热模型")
                from torchvision import transforms
                dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
                transform_temp = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                dummy_tensor = transform_temp(dummy_img).unsqueeze(0)
                with torch.no_grad():
                    _ = model(dummy_tensor)
                logger.info("CPU预热完成，移动到MPS设备")
                model = model.to(self.device)
            else:
                model = model.to(self.device)

            # 创建特征提取hook
            self._efficientnet_features = []
            def hook_fn(module, input, output):
                self._efficientnet_features.append(output)

            # 注册hook到avgpool层，获取1536维特征
            if hasattr(model, 'avgpool'):
                model.avgpool.register_forward_hook(hook_fn)
            else:
                logger.warning("EfficientNet模型没有avgpool层，回退到简单方法")
                return

            # 创建线性投影层 1536 → 512
            # P2-7: 尝试从训练好的权重文件加载投影层
            projection_weights_path = os.path.join(
                os.path.dirname(EFFICIENTNET_MODEL_DIR), "efficientnet_b3", "projection_weights.pth"
            )
            projection = torch.nn.Linear(1536, 512, bias=False)
            if os.path.exists(projection_weights_path):
                try:
                    state_dict = torch.load(projection_weights_path, map_location=self.device)
                    projection.load_state_dict(state_dict)
                    logger.info(f"投影层权重已加载: {projection_weights_path}")
                except Exception as e:
                    logger.warning(f"加载投影层权重失败: {e}，使用 Xavier 初始化")
                    torch.nn.init.xavier_normal_(projection.weight)
            else:
                # P2-7: 权重文件不存在，使用 Xavier 初始化并记录 warning
                torch.nn.init.xavier_normal_(projection.weight)
                logger.warning(
                    "投影层权重文件 models/efficientnet_b3/projection_weights.pth 不存在，"
                    "使用随机 Xavier 初始化。建议训练投影层权重以提升 FAISS 检索质量。"
                )
            projection = projection.to(self.device)
            projection.eval()

            # 创建图像预处理transform
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # 缓存到类级别
            FeatureExtraction._efficientnet_model = model
            FeatureExtraction._efficientnet_projection = projection
            FeatureExtraction._efficientnet_transform = transform

            self.efficientnet_model = model
            self.efficientnet_projection = projection
            self.efficientnet_transform = transform

            # 预热模型
            self._warmup_efficientnet()

            logger.info("EfficientNet-B3特征提取模型加载完成，输出维度: 512")

        except Exception as e:
            logger.error(f"加载EfficientNet模型失败: {e}")
            self.efficientnet_model = None
            self.efficientnet_projection = None
            self.efficientnet_transform = None

    def _warmup_efficientnet(self):
        """预热EfficientNet模型"""
        import torch
        try:
            dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            dummy_tensor = self.efficientnet_transform(dummy_img).unsqueeze(0)
            
            # 如果设备是MPS，预热时使用CPU以避免内存问题
            if self.device == "mps":
                model_cpu = self.efficientnet_model.to("cpu")
                with torch.no_grad():
                    _ = model_cpu(dummy_tensor)
                self.efficientnet_model = model_cpu.to(self.device)
            else:
                dummy_tensor = dummy_tensor.to(self.device)
                with torch.no_grad():
                    _ = self.efficientnet_model(dummy_tensor)
            
            if self._efficientnet_features:
                feat = self._efficientnet_features[-1].squeeze()
                projected = self.efficientnet_projection(feat)
                projected = projected / projected.norm()
            self._efficientnet_features.clear()
            logger.debug("EfficientNet模型预热完成")
        except Exception as e:
            logger.debug(f"EfficientNet模型预热失败: {e}")

    def _warmup_model(self):
        """模型预热，提高首次推理速度"""
        if self.model is None or self.processor is None:
            return
        
        try:
            import torch
            # 创建一个随机的224x224图像作为预热输入
            dummy_input = torch.randn(3, 224, 224).unsqueeze(0)
            if self._use_fp16:
                dummy_input = dummy_input.half()
            dummy_input = dummy_input.to(self.device)
            
            # 运行一次前向传播进行预热
            with torch.no_grad():
                if self.device == "mps":
                    # MPS设备需要特殊处理
                    dummy_pil = Image.new('RGB', (224, 224), color=(0, 0, 0))
                    inputs = self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
                    if self._use_fp16:
                        inputs = {k: v.half() for k, v in inputs.items()}
                    _ = self.model.get_image_features(**inputs)
                else:
                    inputs = self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
                    if self._use_fp16:
                        inputs = {k: v.half() for k, v in inputs.items()}
                    _ = self.model.get_image_features(**inputs)
            
            logger.debug("模型预热完成")
        except Exception as e:
            logger.debug(f"模型预热失败: {e}")

    def extract_features(self, img):
        """提取图像特征"""
        import numpy as np  # 确保在线程池执行时有本地引用
        
        try:
            # 检查输入图像
            if img is None:
                raise ValueError("输入图像为None")

            # 检查输入类型，如果是PyTorch Tensor则转换为PIL Image
            if hasattr(img, "shape") and hasattr(img, "cpu"):
                # 这是一个PyTorch Tensor
                logger.debug("输入是PyTorch Tensor，转换为PIL Image")
                # 如果在GPU上，先移到CPU
                if img.device.type != "cpu":
                    img = img.cpu()
                # 分离计算图（避免 requires_grad=True 时 .numpy() 失败）
                if img.requires_grad:
                    img = img.detach()
                # 转换为numpy数组
                img_array = img.numpy()
                # 去除可能的 batch 维度 (N, C, H, W) -> (C, H, W)
                # 预处理器输出为 4D 张量，extract_features 仅处理单张 3D 张量
                if img_array.ndim == 4:
                    img_array = img_array[0]
                # 如果形状是 (C, H, W)，转换为 (H, W, C)
                if img_array.ndim == 3 and img_array.shape[0] in [1, 3]:
                    img_array = img_array.transpose(1, 2, 0)
                    # 反归一化（ImageNet 标准化），将值恢复到 [0,1] 范围
                    std = [0.229, 0.224, 0.225]
                    mean = [0.485, 0.456, 0.406]
                    img_array = img_array * std + mean
                # 裁剪到合法范围并转换为 PIL Image
                img_array = np.clip(img_array, 0, 1)
                img = Image.fromarray((img_array * 255).astype("uint8"))

            # 优先使用EfficientNet（macOS上的主要方法）
            if self.efficientnet_model is not None and self.efficientnet_projection is not None:
                return self._extract_features_efficientnet(img)
            # 如果CLIP模型已加载，使用CLIP
            elif self.model is not None and self.processor is not None:
                return self._extract_features_clip(img)
            else:
                # 回退到简单特征提取方法
                return self._extract_features_simple(img)

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 抛出异常而非返回随机向量，随机向量会导致误分类
            raise RuntimeError(f"特征提取失败: {e}")

    def _extract_features_efficientnet(self, img):
        """使用EfficientNet-B3模型提取特征"""
        import torch

        try:
            # 预处理图像
            input_tensor = self.efficientnet_transform(img).unsqueeze(0).to(self.device)

            # 清空hook缓存
            self._efficientnet_features = []

            # 推理
            with torch.no_grad():
                _ = self.efficientnet_model(input_tensor)

            # 获取1536维特征
            if not self._efficientnet_features:
                logger.warning("EfficientNet hook未捕获到特征，回退到简单方法")
                return self._extract_features_simple(img)

            feat = self._efficientnet_features[-1].squeeze()  # [1536]

            # 投影到512维
            with torch.no_grad():
                projected = self.efficientnet_projection(feat)  # [512]

            # L2归一化
            norm = projected.norm()
            if norm > 0:
                projected = projected / norm

            features = projected.cpu().numpy().astype(np.float32)

            logger.debug(f"EfficientNet特征提取完成，特征形状: {features.shape}")
            return features

        except Exception as e:
            logger.warning(f"EfficientNet特征提取失败，回退到简单方法: {e}")
            return self._extract_features_simple(img)

    def _extract_features_clip(self, img):
        """使用CLIP模型提取特征"""
        import torch
        
        try:
            # 预处理图像
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            
            # 如果启用FP16，转换为半精度
            if self._use_fp16:
                inputs = {k: v.half() for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            # 归一化并转换为numpy数组
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.squeeze().cpu().numpy().astype(np.float32)
            
            logger.debug(f"CLIP特征提取完成，特征形状: {features.shape}")
            return features
            
        except Exception as e:
            logger.warning(f"CLIP特征提取失败，回退到简单方法: {e}")
            return self._extract_features_simple(img)

    def _extract_features_simple(self, img):
        """简单特征提取方法（降级方案）"""
        logger.debug("使用简单特征提取方法")

        # 调整图像大小
        img = img.resize((224, 224))
        # 转换为numpy数组
        img_array = np.array(img)

        # 提取更丰富的特征
        # 1. 颜色直方图特征 (R, G, B)
        hist_r = np.histogram(img_array[:, :, 0], bins=16, range=(0, 255))[0]
        hist_g = np.histogram(img_array[:, :, 1], bins=16, range=(0, 255))[0]
        hist_b = np.histogram(img_array[:, :, 2], bins=16, range=(0, 255))[0]

        # 2. 纹理特征 (简单的边缘检测)
        from PIL import ImageFilter

        edges = img.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges)
        edge_density = np.mean(edges_array)

        # 3. 颜色统计特征
        mean_color = img_array.mean(axis=(0, 1))
        std_color = img_array.std(axis=(0, 1))

        # 4. 形状特征
        width, height = img.size
        aspect_ratio = width / height

        # 组合所有特征
        features = np.concatenate(
            [
                hist_r,  # 16维
                hist_g,  # 16维
                hist_b,  # 16维
                mean_color,  # 3维
                std_color,  # 3维
                [edge_density, aspect_ratio],  # 2维
            ]
        )

        # 归一化特征
        features = (
            features / np.linalg.norm(features) if np.linalg.norm(features) > 0 else features
        )

        # 填充到512维
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), "constant")
        elif len(features) > 512:
            features = features[:512]

        logger.debug(f"简单特征提取完成，特征形状: {features.shape}")
        return features

    def batch_extract_features(self, imgs, batch_size=8):
        """批量提取图像特征

        Args:
            imgs: 图像列表
            batch_size: 批量大小

        Returns:
            特征向量列表
        """
        try:
            # 检查输入图像列表
            if not imgs:
                return []

            # 如果模型已加载，使用CLIP模型进行批量特征提取
            if self.model is not None and self.processor is not None:
                return self._batch_extract_features_clip(imgs, batch_size)
            else:
                # 使用简单特征提取方法
                logger.debug("使用简单特征提取方法进行批量处理")
                all_features = []
                for img in imgs:
                    feature = self.extract_features(img)
                    all_features.append(feature)

                # 转换为numpy数组
                if all_features:
                    return np.vstack(all_features)
                else:
                    return np.array([])
                    
        except Exception as e:
            logger.error(f"批量特征提取失败: {e}")
            # 发生错误时，返回空数组
            return np.array([])

    def _batch_extract_features_clip(self, imgs, batch_size=8):
        """使用CLIP模型批量提取特征"""
        import torch
        
        try:
            all_features = []
            
            # 分批处理
            for i in range(0, len(imgs), batch_size):
                batch_imgs = imgs[i:i + batch_size]
                
                # 预处理批量图像
                inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
                
                # 如果启用FP16，转换为半精度
                if self._use_fp16:
                    inputs = {k: v.half() for k, v in inputs.items()}
                
                # 推理
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                
                # 归一化并转换为numpy数组
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy().astype(np.float32)
                
                all_features.append(features)
            
            # 合并所有批次的特征
            if all_features:
                return np.vstack(all_features)
            else:
                return np.array([])
                
        except Exception as e:
            logger.warning(f"CLIP批量特征提取失败，回退到简单方法: {e}")
            return self.batch_extract_features(imgs, batch_size)

    def extract_features_from_multiple_characters(self, characters, batch_size=8):
        """从多个角色中提取特征

        Args:
            characters: 角色字典列表
            batch_size: 批量大小

        Returns:
            添加了特征的角色字典列表
        """
        try:
            # 检查输入角色列表
            if not characters:
                return []

            # 提取所有角色的图像
            imgs = [char["image"] for char in characters if "image" in char]

            # 批量提取特征
            features = self.batch_extract_features(imgs, batch_size=batch_size)

            # 将特征与角色信息关联
            feature_idx = 0
            for char in characters:
                if "image" in char:
                    char["feature"] = features[feature_idx]
                    feature_idx += 1

            return characters
        except Exception as e:
            logger.error(f"多角色特征提取失败: {e}")
            return []


if __name__ == "__main__":
    # 测试特征提取模块
    extractor = FeatureExtraction()

    # 测试图像路径（需要根据实际情况修改）
    test_image = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset/aerial_(arknights)/37185069_p0_master1200.jpg"

    try:
        # 加载图像
        img = Image.open(test_image)

        # 提取特征
        features = extractor.extract_features(img)

        logger.info(f"特征向量维度: {features.shape}")
        logger.info(f"特征向量前10个元素: {features[:10]}")
        logger.info("特征提取成功!")
    except Exception as e:
        logger.error(f"测试失败: {e}")
