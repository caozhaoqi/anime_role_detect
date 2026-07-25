"""
链式 QueryHelper 查询构建器
基于 SQLAlchemy 封装的链式查询 API，简化复杂查询的构建

QueryHelper 设计
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from sqlalchemy import or_, and_, asc, desc, func, distinct
from sqlalchemy.orm import Session, Query
from sqlalchemy.sql import expression


class QueryHelper:
    """
    链式查询构建器
    支持 filter, order, paginate, join, group 等操作

    Usage:
        qh = QueryHelper(session, Model)
        results = qh.filter(name="test") \
                   .filter_by(status=1) \
                   .order_by("created_at", desc=True) \
                   .paginate(page=1, size=20) \
                   .all()
    """

    def __init__(self, session: Session, model: Type):
        """
        初始化查询构建器

        Args:
            session: SQLAlchemy Session
            model: SQLAlchemy Model 类
        """
        self.session = session
        self.model = model
        self.query = session.query(model)
        self._filters = []
        self._order_by = []
        self._joins = []
        self._group_by = []
        self._having = []
        self._limit = None
        self._offset = None

    def filter(self, **kwargs) -> 'QueryHelper':
        """
        添加过滤条件（使用 == 匹配）

        Args:
            **kwargs: 过滤条件，如 name="test", status=1

        Returns:
            QueryHelper 实例（链式调用）
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                self._filters.append(getattr(self.model, key) == value)
        return self

    def filter_like(self, **kwargs) -> 'QueryHelper':
        """
        添加模糊匹配条件

        Args:
            **kwargs: 模糊匹配条件，如 name="%test%"

        Returns:
            QueryHelper 实例（链式调用）
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                self._filters.append(getattr(self.model, key).like(value))
        return self

    def filter_in(self, **kwargs) -> 'QueryHelper':
        """
        添加 IN 查询条件

        Args:
            **kwargs: IN 查询条件，如 ids=[1, 2, 3]

        Returns:
            QueryHelper 实例（链式调用）
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                self._filters.append(getattr(self.model, key).in_(value))
        return self

    def filter_not_in(self, **kwargs) -> 'QueryHelper':
        """
        添加 NOT IN 查询条件

        Args:
            **kwargs: NOT IN 查询条件，如 ids=[1, 2, 3]

        Returns:
            QueryHelper 实例（链式调用）
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                self._filters.append(getattr(self.model, key).not_in(value))
        return self

    def filter_greater_than(self, **kwargs) -> 'QueryHelper':
        """
        添加大于条件

        Args:
            **kwargs: 大于条件，如 age=18

        Returns:
            QueryHelper 实例（链式调用）
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                self._filters.append(getattr(self.model, key) > value)
        return self

    def filter_less_than(self, **kwargs) -> 'QueryHelper':
        """
        添加小于条件

        Args:
            **kwargs: 小于条件，如 age=60

        Returns:
            QueryHelper 实例（链式调用）
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                self._filters.append(getattr(self.model, key) < value)
        return self

    def filter_between(self, **kwargs) -> 'QueryHelper':
        """
        添加 BETWEEN 条件

        Args:
            **kwargs: BETWEEN 条件，如 age=(18, 60)

        Returns:
            QueryHelper 实例（链式调用）
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key) and isinstance(value, tuple):
                self._filters.append(getattr(self.model, key).between(*value))
        return self

    def filter_is_null(self, *columns) -> 'QueryHelper':
        """
        添加 IS NULL 条件

        Args:
            *columns: 列名列表

        Returns:
            QueryHelper 实例（链式调用）
        """
        for column in columns:
            if hasattr(self.model, column):
                self._filters.append(getattr(self.model, column).is_(None))
        return self

    def filter_is_not_null(self, *columns) -> 'QueryHelper':
        """
        添加 IS NOT NULL 条件

        Args:
            *columns: 列名列表

        Returns:
            QueryHelper 实例（链式调用）
        """
        for column in columns:
            if hasattr(self.model, column):
                self._filters.append(getattr(self.model, column).isnot(None))
        return self

    def filter_or(self, *conditions) -> 'QueryHelper':
        """
        添加 OR 条件组

        Args:
            *conditions: 条件表达式，如 (model.name == "test")

        Returns:
            QueryHelper 实例（链式调用）
        """
        self._filters.append(or_(*conditions))
        return self

    def filter_and(self, *conditions) -> 'QueryHelper':
        """
        添加 AND 条件组

        Args:
            *conditions: 条件表达式，如 (model.name == "test")

        Returns:
            QueryHelper 实例（链式调用）
        """
        self._filters.append(and_(*conditions))
        return self

    def order_by(self, column: str, descending: bool = False) -> 'QueryHelper':
        """
        添加排序条件

        Args:
            column: 列名
            descending: 是否降序

        Returns:
            QueryHelper 实例（链式调用）
        """
        if hasattr(self.model, column):
            col = getattr(self.model, column)
            self._order_by.append(desc(col) if descending else asc(col))
        return self

    def join(self, join_model: Type, on_condition=None, isouter: bool = False) -> 'QueryHelper':
        """
        添加 JOIN

        Args:
            join_model: 关联模型
            on_condition: JOIN 条件，如 model.id == other_model.parent_id
            isouter: 是否左外连接

        Returns:
            QueryHelper 实例（链式调用）
        """
        self._joins.append((join_model, on_condition, isouter))
        return self

    def group_by(self, *columns) -> 'QueryHelper':
        """
        添加 GROUP BY

        Args:
            *columns: 列名列表

        Returns:
            QueryHelper 实例（链式调用）
        """
        for column in columns:
            if hasattr(self.model, column):
                self._group_by.append(getattr(self.model, column))
        return self

    def having(self, condition) -> 'QueryHelper':
        """
        添加 HAVING 条件

        Args:
            condition: HAVING 条件表达式

        Returns:
            QueryHelper 实例（链式调用）
        """
        self._having.append(condition)
        return self

    def _apply_filters(self) -> None:
        """应用所有过滤条件"""
        if self._filters:
            self.query = self.query.filter(and_(*self._filters))

    def _apply_order_by(self) -> None:
        """应用排序条件"""
        if self._order_by:
            self.query = self.query.order_by(*self._order_by)

    def _apply_joins(self) -> None:
        """应用 JOIN"""
        for join_model, on_condition, isouter in self._joins:
            if on_condition:
                self.query = self.query.join(join_model, on_condition, isouter=isouter)
            else:
                self.query = self.query.join(join_model, isouter=isouter)

    def _apply_group_by(self) -> None:
        """应用 GROUP BY"""
        if self._group_by:
            self.query = self.query.group_by(*self._group_by)

    def _apply_having(self) -> None:
        """应用 HAVING"""
        if self._having:
            self.query = self.query.having(and_(*self._having))

    def _apply_limit_offset(self) -> None:
        """应用 LIMIT 和 OFFSET（必须放在最后）"""
        if self._offset is not None:
            self.query = self.query.offset(self._offset)
        if self._limit is not None:
            self.query = self.query.limit(self._limit)

    def _build_query(self) -> Query:
        """构建最终查询"""
        self._apply_joins()
        self._apply_filters()
        self._apply_group_by()
        self._apply_having()
        self._apply_order_by()
        self._apply_limit_offset()
        return self.query

    def all(self) -> List[Any]:
        """执行查询，返回所有结果"""
        return self._build_query().all()

    def first(self) -> Optional[Any]:
        """执行查询，返回第一个结果"""
        return self._build_query().first()

    def one(self) -> Any:
        """执行查询，返回唯一结果"""
        return self._build_query().one()

    def one_or_none(self) -> Optional[Any]:
        """执行查询，返回唯一结果或 None"""
        return self._build_query().one_or_none()

    def count(self) -> int:
        """执行查询，返回结果数量"""
        return self._build_query().count()

    def scalar(self) -> Any:
        """执行查询，返回标量值"""
        return self._build_query().scalar()

    def exists(self) -> bool:
        """检查是否存在匹配结果"""
        query = self._build_query()
        return self.session.query(query.exists()).scalar()

    def paginate(self, page: int = 1, size: int = 20) -> Tuple[List[Any], int]:
        """
        分页查询

        Args:
            page: 页码（从1开始）
            size: 每页大小

        Returns:
            (结果列表, 总数量)
        """
        if self._limit is not None or self._offset is not None:
            query = self._build_query()
            total = query.count()
            items = query.all()
        else:
            base_query = self._build_query_without_limit_offset()
            total = base_query.count()
            offset = (page - 1) * size
            items = base_query.offset(offset).limit(size).all()
        return items, total

    def _build_query_without_limit_offset(self) -> Query:
        """构建查询但不应用 LIMIT 和 OFFSET"""
        self._apply_joins()
        self._apply_filters()
        self._apply_group_by()
        self._apply_having()
        self._apply_order_by()
        return self.query

    def limit(self, limit: int) -> 'QueryHelper':
        """
        添加 LIMIT

        Args:
            limit: 限制数量

        Returns:
            QueryHelper 实例（链式调用）
        """
        self._limit = limit
        return self

    def offset(self, offset: int) -> 'QueryHelper':
        """
        添加 OFFSET

        Args:
            offset: 偏移量

        Returns:
            QueryHelper 实例（链式调用）
        """
        self._offset = offset
        return self

    def distinct(self) -> 'QueryHelper':
        """
        添加 DISTINCT

        Returns:
            QueryHelper 实例（链式调用）
        """
        self.query = self.query.distinct()
        return self

    def get_query(self) -> Query:
        """获取构建后的 Query 对象"""
        return self._build_query()


class QueryBuilder:
    """
    查询构建器（静态工厂方法）
    提供便捷的查询入口

    Usage:
        from core.query_helper import QueryBuilder

        # 基本查询
        results = QueryBuilder(session, Model).filter(name="test").all()

        # 统计查询
        count = QueryBuilder(session, Model).filter(status=1).count()

        # 分页查询
        items, total = QueryBuilder(session, Model).paginate(page=1, size=20)
    """

    @classmethod
    def query(cls, session: Session, model: Type) -> QueryHelper:
        """创建查询构建器"""
        return QueryHelper(session, model)

    @classmethod
    def get_by_id(cls, session: Session, model: Type, id: Any) -> Optional[Any]:
        """根据 ID 获取单条记录"""
        return session.query(model).get(id)

    @classmethod
    def get_one(cls, session: Session, model: Type, **kwargs) -> Optional[Any]:
        """根据条件获取单条记录"""
        return QueryHelper(session, model).filter(**kwargs).first()

    @classmethod
    def get_list(cls, session: Session, model: Type, **kwargs) -> List[Any]:
        """根据条件获取记录列表"""
        return QueryHelper(session, model).filter(**kwargs).all()

    @classmethod
    def get_paginated(cls, session: Session, model: Type, page: int = 1, size: int = 20, **kwargs) -> Tuple[List[Any], int]:
        """分页获取记录"""
        return QueryHelper(session, model).filter(**kwargs).paginate(page=page, size=size)

    @classmethod
    def exists(cls, session: Session, model: Type, **kwargs) -> bool:
        """检查是否存在记录"""
        return QueryHelper(session, model).filter(**kwargs).exists()

    @classmethod
    def count(cls, session: Session, model: Type, **kwargs) -> int:
        """统计记录数量"""
        return QueryHelper(session, model).filter(**kwargs).count()