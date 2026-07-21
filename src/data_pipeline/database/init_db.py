"""
数据库初始化脚本
Database Initialization Script
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone

Base = declarative_base()


class Character(Base):
    """角色表"""
    __tablename__ = 'characters'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    series = Column(String(100), nullable=False, index=True)
    aliases = Column(JSON, nullable=True)
    search_terms = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    # 关系
    samples = relationship("Sample", back_populates="character", cascade="all, delete-orphan")
    collection_tasks = relationship("CollectionTask", back_populates="character", cascade="all, delete-orphan")


class Sample(Base):
    """样本表"""
    __tablename__ = 'samples'

    id = Column(Integer, primary_key=True)
    image_path = Column(String(500), nullable=False, unique=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=True, index=True)

    # 质量指标
    quality_score = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    # 分类标签
    is_anime = Column(Boolean, nullable=True)
    is_ai_generated = Column(Boolean, nullable=True)
    anime_confidence = Column(Float, nullable=True)
    ai_confidence = Column(Float, nullable=True)

    # 检测信息
    person_count = Column(Integer, nullable=True)
    bbox_area_ratio = Column(Float, nullable=True)

    # 属性标签
    attributes = Column(JSON, nullable=True)

    # 推理信息
    confidence = Column(Float, nullable=True)
    is_difficult = Column(Boolean, default=False)

    # 状态管理
    status = Column(String(20), default='pending', index=True)  # pending, reviewed, rejected
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    # 关系
    character = relationship("Character", back_populates="samples")


class CollectionTask(Base):
    """采集任务表"""
    __tablename__ = 'collection_tasks'

    id = Column(Integer, primary_key=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=True, index=True)
    search_terms = Column(JSON, nullable=True)
    max_samples = Column(Integer, default=1000)
    status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed
    collected_count = Column(Integer, default=0)
    error_message = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # 关系
    character = relationship("Character", back_populates="collection_tasks")


class DeduplicationRecord(Base):
    """去重记录表"""
    __tablename__ = 'deduplication_records'

    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False, index=True)
    duplicate_of_id = Column(Integer, ForeignKey('samples.id'), nullable=True)
    similarity = Column(Float, nullable=True)
    method = Column(String(50), nullable=False)  # phash, clip
    created_at = Column(DateTime, default=datetime.now(timezone.utc))


class Annotation(Base):
    """标注记录表"""
    __tablename__ = 'annotations'

    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False, index=True)
    annotator = Column(String(50), nullable=False)  # auto, human
    bbox = Column(JSON, nullable=True)  # [x1, y1, x2, y2]
    confidence = Column(Float, nullable=True)
    character_name = Column(String(100), nullable=True, index=True)  # 识别的角色名
    character_confidence = Column(Float, nullable=True)  # 角色识别置信度
    attributes = Column(JSON, nullable=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))


def init_database(db_url: str = "sqlite:///./data/data_pipeline.db"):
    """初始化数据库"""
    print(f"正在初始化数据库: {db_url}")

    # 创建引擎
    engine = create_engine(
        db_url,
        pool_size=20,
        max_overflow=50,
        echo=False
    )

    # 创建所有表
    Base.metadata.create_all(engine)

    # 创建会话工厂
    Session = sessionmaker(bind=engine)

    print("✅ 数据库初始化完成！")
    print(f"📊 已创建表: {', '.join(Base.metadata.tables.keys())}")

    return engine, Session


def load_character_aliases(db_session, aliases_file: str = "data/character_aliases.json"):
    """加载角色别名到数据库"""
    import json
    from pathlib import Path

    aliases_path = Path(aliases_file)
    if not aliases_path.exists():
        print(f"⚠️ 角色别名文件不存在: {aliases_file}")
        return

    with open(aliases_path, 'r', encoding='utf-8') as f:
        aliases_data = json.load(f)

    loaded_count = 0
    for key, data in aliases_data.items():
        # 检查是否已存在
        existing = db_session.query(Character).filter_by(name=data['name']).first()
        if existing:
            continue

        character = Character(
            id=data['id'],
            name=data['name'],
            series=data['series'],
            aliases=data.get('aliases', []),
            search_terms=data.get('search_terms', [])
        )
        db_session.add(character)
        loaded_count += 1

    db_session.commit()
    print(f"✅ 已加载 {loaded_count} 个角色到数据库")


if __name__ == "__main__":
    # 初始化数据库
    engine, Session = init_database()

    # 加载角色别名
    session = Session()
    load_character_aliases(session)

    # 关闭会话
    session.close()
    engine.dispose()

    print("\n🎉 数据库初始化完成！")
