"""
BaseModel 基类增强
BaseModel 设计，添加声明式配置支持

功能：
1. __info_dict__ - 关联对象自动翻译
2. __blur_field__ - 声明式模糊搜索字段
3. __flex_field__ - 弹性扩展字段
4. to_dict() - 统一的字典转换方法
5. from_dict() - 从字典创建对象
6. copy() - 对象复制
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

try:
    from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float, Text
    from sqlalchemy.ext.declarative import declarative_base, AbstractConcreteBase
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    Base = object
    Column = None
    Integer = None
    String = None
    DateTime = None
    Boolean = None
    JSON = None
    Float = None
    Text = None


T = TypeVar('T', bound='EnhancedBase')


class EnhancedBase:
    """
    增强型 Model 基类
    提供声明式配置和便捷方法

    声明式属性：
    - __description__: 模型描述
    - __info_dict__: 关联对象自动翻译配置
      格式: [(origin_field, target_field, ObjectClass), ...]
      示例: [('department_id', 'department', Department)]
    - __blur_field__: 模糊搜索字段列表
      格式: ['field1', 'field2', ...]
      示例: ['name', 'email', 'username']
    - __flex_field__: 弹性扩展字段名（JSON类型）
      默认: 'flex_data'
    - __ignore_fields__: to_dict() 时忽略的字段列表
      示例: ['password_hash', 'token']
    - __display_field__: 显示字段（用于日志、展示）
      默认: 'id'
    """

    __description__: Optional[str] = None
    __info_dict__: Optional[List[Tuple[str, str, Type]]] = None
    __blur_field__: Optional[List[str]] = None
    __flex_field__: str = 'flex_data'
    __ignore_fields__: Optional[List[str]] = None
    __display_field__: str = 'id'

    def __repr__(self):
        """对象表示"""
        display_value = getattr(self, self.__display_field__, 'N/A')
        return f'<{self.__class__.__name__}:{display_value}>'

    def to_dict(self, include_info: bool = False) -> Dict[str, Any]:
        """
        转换为字典

        Args:
            include_info: 是否包含关联对象翻译

        Returns:
            字典表示
        """
        result = {}
        ignore_fields = self.__ignore_fields__ or []
        ignore_fields = set(ignore_fields + ['_sa_instance_state'])

        for attr in self.__dict__:
            if attr.startswith('_'):
                continue
            if attr in ignore_fields:
                continue

            try:
                value = getattr(self, attr)
            except Exception:
                continue

            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, bytes):
                value = value.decode('utf-8', errors='replace')
            elif isinstance(value, (list, dict)):
                try:
                    value = json.loads(json.dumps(value))
                except Exception:
                    pass

            result[attr] = value

        if include_info and self.__info_dict__:
            self._fill_info_dict(result)

        return result

    def _fill_info_dict(self, result: Dict[str, Any]) -> None:
        """
        填充关联对象翻译

        Args:
            result: 目标字典
        """
        if not self.__info_dict__:
            return

        for origin_field, target_field, obj_class in self.__info_dict__:
            origin_value = getattr(self, origin_field, None)
            if origin_value is None:
                continue

            try:
                info_obj = obj_class.get_by_id(origin_value)
                if info_obj:
                    result[target_field] = info_obj.to_dict()
            except Exception:
                pass

    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        从字典创建对象

        Args:
            data: 字典数据

        Returns:
            对象实例
        """
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                if isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value)
                    except Exception:
                        pass
                setattr(instance, key, value)
        return instance

    def update_from_dict(self, data: Dict[str, Any], ignore_fields: Optional[List[str]] = None) -> None:
        """
        从字典更新对象

        Args:
            data: 字典数据
            ignore_fields: 忽略的字段列表
        """
        ignore = set(ignore_fields or [])
        ignore.add('id')

        for key, value in data.items():
            if key in ignore:
                continue
            if hasattr(self, key):
                if isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value)
                    except Exception:
                        pass
                setattr(self, key, value)

    def copy(self: T, exclude_fields: Optional[List[str]] = None) -> T:
        """
        复制对象

        Args:
            exclude_fields: 排除的字段列表

        Returns:
            对象副本
        """
        exclude = set(exclude_fields or [])
        exclude.add('id')

        data = self.to_dict()
        for field in exclude:
            data.pop(field, None)

        return self.from_dict(data)

    def get_flex_data(self, key: str, default: Any = None) -> Any:
        """
        获取弹性扩展字段值

        Args:
            key: 弹性字段键
            default: 默认值

        Returns:
            弹性字段值
        """
        flex_field = getattr(self, self.__flex_field__, None)
        if flex_field is None:
            return default

        if isinstance(flex_field, str):
            try:
                flex_data = json.loads(flex_field)
            except Exception:
                return default
        elif isinstance(flex_field, dict):
            flex_data = flex_field
        else:
            return default

        return flex_data.get(key, default)

    def set_flex_data(self, key: str, value: Any) -> None:
        """
        设置弹性扩展字段值

        Args:
            key: 弹性字段键
            value: 弹性字段值
        """
        flex_field = getattr(self, self.__flex_field__, None)

        if isinstance(flex_field, str):
            try:
                flex_data = json.loads(flex_field)
            except Exception:
                flex_data = {}
        elif isinstance(flex_field, dict):
            flex_data = flex_field
        else:
            flex_data = {}

        flex_data[key] = value
        setattr(self, self.__flex_field__, json.dumps(flex_data))

    def clear_flex_data(self) -> None:
        """清空弹性扩展字段"""
        setattr(self, self.__flex_field__, json.dumps({}))

    @classmethod
    def get_blur_fields(cls) -> List[str]:
        """
        获取模糊搜索字段列表

        Returns:
            模糊搜索字段列表
        """
        return cls.__blur_field__ or []

    @classmethod
    def get_info_dict(cls) -> Optional[List[Tuple[str, str, Type]]]:
        """
        获取关联对象配置

        Returns:
            关联对象配置
        """
        return cls.__info_dict__

    @classmethod
    def get_description(cls) -> Optional[str]:
        """
        获取模型描述

        Returns:
            模型描述
        """
        return cls.__description__


if HAS_SQLALCHEMY:
    _Base = declarative_base()

    class BaseModel(AbstractConcreteBase, _Base, EnhancedBase):
        """
        SQLAlchemy 增强型 BaseModel
        继承自 EnhancedBase，提供声明式配置和便捷方法
        """

        __abstract__ = True

        id = Column(Integer, primary_key=True, index=True)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

else:
    class BaseModel(EnhancedBase):
        """
        非 SQLAlchemy 环境下的 BaseModel
        """

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class ModelMixin:
    """
    模型混合类
    提供通用的模型操作方法
    """

    @classmethod
    def get_by_id(cls, session, id: Any) -> Optional['ModelMixin']:
        """
        根据 ID 获取记录

        Args:
            session: 数据库会话
            id: 记录 ID

        Returns:
            记录对象或 None
        """
        if session is None:
            return None
        return session.query(cls).get(id)

    @classmethod
    def get_one(cls, session, **kwargs) -> Optional['ModelMixin']:
        """
        根据条件获取单条记录

        Args:
            session: 数据库会话
            **kwargs: 过滤条件

        Returns:
            记录对象或 None
        """
        if session is None:
            return None
        return session.query(cls).filter_by(**kwargs).first()

    @classmethod
    def get_list(cls, session, **kwargs) -> List['ModelMixin']:
        """
        根据条件获取记录列表

        Args:
            session: 数据库会话
            **kwargs: 过滤条件

        Returns:
            记录列表
        """
        if session is None:
            return []
        return session.query(cls).filter_by(**kwargs).all()

    @classmethod
    def exists(cls, session, **kwargs) -> bool:
        """
        检查是否存在记录

        Args:
            session: 数据库会话
            **kwargs: 过滤条件

        Returns:
            是否存在
        """
        if session is None:
            return False
        return session.query(cls).filter_by(**kwargs).first() is not None

    @classmethod
    def count(cls, session, **kwargs) -> int:
        """
        统计记录数量

        Args:
            session: 数据库会话
            **kwargs: 过滤条件

        Returns:
            记录数量
        """
        if session is None:
            return 0
        return session.query(cls).filter_by(**kwargs).count()

    def save(self, session) -> None:
        """
        保存记录

        Args:
            session: 数据库会话
        """
        if session is None:
            return
        session.add(self)
        session.commit()

    def delete(self, session) -> None:
        """
        删除记录

        Args:
            session: 数据库会话
        """
        if session is None:
            return
        session.delete(self)
        session.commit()