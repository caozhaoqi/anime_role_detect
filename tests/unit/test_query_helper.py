"""
QueryHelper 回归测试
https://github.com/sqlalchemy/sqlalchemy

验证在多调用终结方法（count/all/first 等）场景下不会触发 SQLAlchemy 2.0 的
"Query.filter() being called on a Query object that has already invoked
limit() or offset()" 报错。

背景：历史日志 07-25 在 database_service.py:106 (RecognitionRecordDB.get_by_user)
抛出该错误，根因是 QueryHelper._build_query() 原地改写 self.query，导致同一实例
第二次构建时在已带 limit/offset 的 query 上执行 filter() 而报错。
修复后每次构建都从干净的 base query 开始。
"""
import pytest
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

from src.core.query_helper import QueryHelper


Base = declarative_base()


class User(Base):
    __tablename__ = "regression_user"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    s.add_all([
        User(name="alice", age=20),
        User(name="bob", age=30),
        User(name="carol", age=40),
        User(name="dave", age=50),
    ])
    s.commit()
    yield s
    s.close()


def test_repeated_count_then_all_on_same_instance(session):
    """回归核心：旧代码在第二次构建时抛 'filter after limit/offset'。"""
    qh = QueryHelper(session, User).filter(name="alice").limit(5)
    count = qh.count()          # 第一次构建
    rows = qh.all()             # 第二次构建（旧代码此处报错）
    assert count == 1
    assert len(rows) == 1
    assert rows[0].name == "alice"


def test_repeated_first_then_all_on_same_instance(session):
    """同一实例先 first 再 all，旧代码同样会在第二次构建时报错。"""
    qh = QueryHelper(session, User).order_by("age").limit(10)
    first = qh.first()          # 第一次构建
    rows = qh.all()             # 第二次构建
    assert first.name == "alice"
    assert len(rows) == 4


def test_filter_offset_limit_all_returns_correct_subset(session):
    """get_by_user 式 filter + order_by + offset + limit + all。"""
    qh = QueryHelper(session, User).order_by("age", descending=True).offset(1).limit(2)
    rows = qh.all()
    names = [r.name for r in rows]
    assert names == ["carol", "bob"]   # 降序后跳过 dave(50)，取 carol(40), bob(30)


def test_paginate_works(session):
    """paginate 不应受重复构建影响。"""
    qh = QueryHelper(session, User).order_by("age")
    items, total = qh.paginate(page=1, size=2)
    assert total == 4
    assert len(items) == 2
    assert items[0].name == "alice"


def test_multiple_independent_instances(session):
    """多个独立实例各自调用终结方法互不影响。"""
    qh_a = QueryHelper(session, User).filter(age=20)
    qh_b = QueryHelper(session, User).filter(age=30)
    assert qh_a.count() == 1
    assert qh_b.count() == 1
    assert qh_a.all()[0].name == "alice"
    assert qh_b.all()[0].name == "bob"
