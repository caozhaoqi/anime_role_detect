#!/usr/bin/env python3
import os
import re

def fix_imports_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复导入路径
    # from core... -> from src.core...
    content = re.sub(r'^from core\.', r'from src.core.', content, flags=re.MULTILINE)
    content = re.sub(r'^import core\.', r'import src.core.', content, flags=re.MULTILINE)
    
    # from utils... -> from src.utils...
    content = re.sub(r'^from utils\.', r'from src.utils.', content, flags=re.MULTILINE)
    content = re.sub(r'^import utils\.', r'import src.utils.', content, flags=re.MULTILINE)
    
    # from models... -> from src.models...
    content = re.sub(r'^from models\.', r'from src.models.', content, flags=re.MULTILINE)
    content = re.sub(r'^import models\.', r'import src.models.', content, flags=re.MULTILINE)
    
    # from data... -> from src.data...
    content = re.sub(r'^from data\.', r'from src.data.', content, flags=re.MULTILINE)
    content = re.sub(r'^import data\.', r'import src.data.', content, flags=re.MULTILINE)
    
    # from config... -> from src.config...
    content = re.sub(r'^from config\.', r'from src.config.', content, flags=re.MULTILINE)
    content = re.sub(r'^import config\.', r'import src.config.', content, flags=re.MULTILINE)
    
    # from backend... -> from src.backend...
    content = re.sub(r'^from backend\.', r'from src.backend.', content, flags=re.MULTILINE)
    content = re.sub(r'^import backend\.', r'import src.backend.', content, flags=re.MULTILINE)
    
    # from web... -> from src.web...
    content = re.sub(r'^from web\.', r'from src.web.', content, flags=re.MULTILINE)
    content = re.sub(r'^import web\.', r'import src.web.', content, flags=re.MULTILINE)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {file_path}")
        return True
    return False

def main():
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    updated_count = 0
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_imports_in_file(file_path):
                    updated_count += 1
    
    print(f"\nTotal files updated: {updated_count}")

if __name__ == "__main__":
    main()
