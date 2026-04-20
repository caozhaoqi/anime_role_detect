#!/usr/bin/env python3
"""
测试validate_image函数
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_validate_image():
    from src.backend.api.app import validate_image
    from fastapi import UploadFile
    from io import BytesIO
    
    # 测试SVG文件
    svg_path = os.path.join(project_root, 'data', 'train', '日奈', '日奈_1.svg')
    with open(svg_path, 'rb') as f:
        svg_content = f.read()
    
    # 创建UploadFile对象
    class MockUploadFile:
        def __init__(self, content, content_type, filename):
            self.file = BytesIO(content)
            self.content_type = content_type
            self.filename = filename
        
        async def read(self):
            return self.file.getvalue()
    
    # 测试SVG文件
    print("测试SVG文件...")
    try:
        file = MockUploadFile(svg_content, 'image/svg+xml', '日奈_1.svg')
        temp_path = validate_image(file, svg_content)
        print(f"SVG文件验证成功，临时文件路径: {temp_path}")
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("临时文件已删除")
    except Exception as e:
        print(f"SVG文件验证失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试完成")

if __name__ == "__main__":
    test_validate_image()
