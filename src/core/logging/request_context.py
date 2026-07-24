import threading
from typing import Any, Optional, Dict
from uuid import uuid4


class RequestContext:
    _local = threading.local()

    @classmethod
    def _get_context(cls) -> dict:
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        return cls._local.context

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        ctx = cls._get_context()
        return ctx.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> Any:
        ctx = cls._get_context()
        ctx[key] = value
        return value

    @classmethod
    def set_many(cls, items: Dict[str, Any]) -> None:
        ctx = cls._get_context()
        ctx.update(items)

    @classmethod
    def delete(cls, key: str) -> None:
        ctx = cls._get_context()
        ctx.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        if hasattr(cls._local, 'context'):
            cls._local.context = {}

    @classmethod
    def exists(cls, key: str) -> bool:
        ctx = cls._get_context()
        return key in ctx

    @classmethod
    def keys(cls) -> list:
        ctx = cls._get_context()
        return list(ctx.keys())

    @classmethod
    def get_trace_id(cls) -> str:
        trace_id = cls.get('trace_id')
        if trace_id is None:
            trace_id = str(uuid4())
            cls.set('trace_id', trace_id)
        return trace_id

    @classmethod
    def get_request_id(cls) -> str:
        request_id = cls.get('request_id')
        if request_id is None:
            request_id = str(uuid4())[:8]
            cls.set('request_id', request_id)
        return request_id

    @classmethod
    def get_user_id(cls) -> Optional[str]:
        return cls.get('user_id')

    @classmethod
    def get_span_id(cls) -> Optional[str]:
        return cls.get('span_id')

    @classmethod
    def set_trace_context(cls, trace_id: str, span_id: str) -> None:
        cls.set('trace_id', trace_id)
        cls.set('span_id', span_id)

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return cls._get_context().copy()


class RequestContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        RequestContext.clear()


def with_request_context(**kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs_inner):
            with RequestContextManager():
                RequestContext.set_many(kwargs)
                return func(*args, **kwargs_inner)
        return wrapper
    return decorator
