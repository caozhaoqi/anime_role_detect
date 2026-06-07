"""重新导入数据到数据库"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.database.init_db import init_database, Character, Sample

engine, Session = init_database()
session = Session()

data_dir = Path('data/danbooru_images')
imported_chars = 0
imported_samples = 0

for char_dir in sorted(data_dir.iterdir()):
    if not char_dir.is_dir():
        continue
    
    dir_name = char_dir.name
    if '_(' in dir_name:
        name = dir_name.split('_(')[0]
        series = dir_name.split('_(')[1].rstrip(')')
    else:
        name = dir_name
        series = 'unknown'
    
    char = session.query(Character).filter_by(name=name).first()
    if not char:
        char = Character(name=name, series=series)
        session.add(char)
        session.flush()
        imported_chars += 1
    
    for img_file in char_dir.rglob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            sample = Sample(image_path=str(img_file), character_id=char.id, status='pending')
            session.add(sample)
            imported_samples += 1
    
    print(f'已处理: {char_dir.name} ({imported_samples} samples)')
    session.commit()

session.commit()
print(f'\n✅ 完成: {imported_chars} characters, {imported_samples} samples')
session.close()