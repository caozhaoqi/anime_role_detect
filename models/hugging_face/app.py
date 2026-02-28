import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image

# 1. Load the checkpoint dictionary
checkpoint = torch.load("model_best.pth", map_location="cpu", weights_only=False)

# 2. Extract labels from the checkpoint's class_to_idx
if "class_to_idx" in checkpoint:
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    LABELS = [idx_to_class[i] for i in range(len(idx_to_class))]
else:
    LABELS = ["Class_0", "Class_1", "Class_2"] 

num_classes = len(LABELS)

# ==========================================
# 3. CRITICAL FIX: Initialize ResNet instead of MobileNetV2
# ==========================================
# 我们使用 resnet18 (如果你训练的是 resnet34 或 resnet50，请对应修改这里)
model = models.resnet18(weights=None) 
# ResNet 的最后一层叫 fc (fully connected)，而不是 classifier
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
# ==========================================

# 4. Load the actual model weights (model_state_dict)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 5. Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 6. Define the prediction function
def predict(image):
    image = image.convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=1)[0] # 修正 dim 为 1
    
    return {LABELS[i]: float(probabilities[i]) for i in range(num_classes)}

# 7. Build the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5), 
    title="Anime Role Classifier (ResNet)",
    description="上传一张动漫图片，模型将识别其角色。（当前加载的权重为 ResNet 架构）"
)

if __name__ == "__main__":
    interface.launch()