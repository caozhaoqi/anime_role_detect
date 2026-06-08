#!/usr/bin/env python3
"""NSFW检测命令行接口"""

import os
import sys
import json

# 在导入任何库之前设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "请提供图片路径"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from PIL import Image
        
        model_name = "Falconsai/nsfw_image_detection"
        LABELS = ["drawings", "hentai", "neutral", "porn", "sexy"]
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        details = {}
        for i, label in enumerate(LABELS):
            details[label] = float(probabilities[0][i])
        
        max_score = float(probabilities.max())
        max_index = int(probabilities.argmax())
        predicted_label = LABELS[max_index]
        
        nsfw_categories = ["porn", "sexy", "hentai"]
        nsfw_score = sum(details.get(cat, 0) for cat in nsfw_categories)
        is_nsfw = nsfw_score > 0.5
        skin_ratio = details.get("sexy", 0) * 0.6 + details.get("porn", 0) * 0.4
        
        result = {
            "is_nsfw": is_nsfw,
            "skin_ratio": float(skin_ratio),
            "nsfw_score": float(nsfw_score),
            "details": details,
            "method": "transformers",
            "predicted_label": predicted_label,
            "confidence": max_score
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()