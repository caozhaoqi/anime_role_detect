from PIL import Image, ImageDraw, ImageFont
import os

# 创建测试图像目录
os.makedirs('test_images', exist_ok=True)

# 创建一个简单的测试图像
img = Image.new('RGB', (400, 300), color='white')
d = ImageDraw.Draw(img)
d.text((50, 130), 'Test Image', fill='black')

# 保存测试图像
img.save('test_images/test1.jpg')
print("测试图像已创建: test_images/test1.jpg")
