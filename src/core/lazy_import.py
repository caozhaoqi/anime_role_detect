"""
延迟加载模块
延迟加载设计，避免启动时加载可选依赖

功能：
1. lazy_import - 延迟导入装饰器
2. LazyModule - 延迟加载模块
3. check_dependency - 检查依赖是否可用
4. 常用依赖的延迟加载封装
"""

import importlib
import logging
from typing import Any, Callable, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LazyImportError(Exception):
    """延迟导入异常"""
    pass


def lazy_import(module_name: str, attr_name: Optional[str] = None, fallback: Optional[Any] = None):
    """
    延迟导入模块或属性

    Args:
        module_name: 模块名
        attr_name: 属性名（可选）
        fallback: 降级值（可选）

    Returns:
        延迟加载对象

    Usage:
        # 延迟导入整个模块
        torch = lazy_import('torch')
        # 使用时才真正导入
        result = torch.tensor([1, 2, 3])

        # 延迟导入模块属性
        numpy = lazy_import('numpy')
        array = lazy_import('numpy', 'array')
        result = array([1, 2, 3])

        # 带降级值的延迟导入
        optional_module = lazy_import('optional_module', fallback=None)
        if optional_module is not None:
            optional_module.do_something()
    """

    class LazyObject:
        _module = None
        _attr = None

        def __getattr__(self, name):
            if self._module is None:
                try:
                    self._module = importlib.import_module(module_name)
                    if attr_name:
                        self._attr = getattr(self._module, attr_name)
                except ImportError as e:
                    if fallback is not None:
                        logger.warning(f"无法导入 {module_name}{'.' + attr_name if attr_name else ''}: {e}，使用降级值")
                        return getattr(fallback, name, None)
                    raise LazyImportError(f"无法导入 {module_name}{'.' + attr_name if attr_name else ''}: {e}")

            if self._attr is not None:
                return getattr(self._attr, name)
            return getattr(self._module, name)

        def __call__(self, *args, **kwargs):
            if self._module is None:
                try:
                    self._module = importlib.import_module(module_name)
                    if attr_name:
                        self._attr = getattr(self._module, attr_name)
                except ImportError as e:
                    if fallback is not None:
                        logger.warning(f"无法导入 {module_name}{'.' + attr_name if attr_name else ''}: {e}，使用降级值")
                        return fallback(*args, **kwargs) if callable(fallback) else fallback
                    raise LazyImportError(f"无法导入 {module_name}{'.' + attr_name if attr_name else ''}: {e}")

            if self._attr is not None:
                return self._attr(*args, **kwargs)
            return self._module(*args, **kwargs)

        def __bool__(self):
            if self._module is not None:
                return True
            try:
                self._module = importlib.import_module(module_name)
                return True
            except ImportError:
                return fallback is not None

        def __repr__(self):
            if self._module is not None:
                if self._attr is not None:
                    return repr(self._attr)
                return repr(self._module)
            return f"<LazyImport: {module_name}{'.' + attr_name if attr_name else ''}>"

    return LazyObject()


class LazyModule:
    """
    延迟加载模块包装器
    支持批量延迟导入

    Usage:
        lazy = LazyModule()
        torch = lazy.import_module('torch')
        numpy = lazy.import_module('numpy')
        array = lazy.import_attr('numpy', 'array')

        # 检查是否可用
        if lazy.is_available('torch'):
            torch.tensor([1, 2, 3])
    """

    def __init__(self):
        self._modules = {}
        self._attrs = {}
        self._checked = {}

    def import_module(self, module_name: str, fallback: Optional[Any] = None) -> Any:
        """
        延迟导入模块

        Args:
            module_name: 模块名
            fallback: 降级值

        Returns:
            延迟加载对象
        """
        if module_name not in self._modules:
            self._modules[module_name] = lazy_import(module_name, fallback=fallback)
        return self._modules[module_name]

    def import_attr(self, module_name: str, attr_name: str, fallback: Optional[Any] = None) -> Any:
        """
        延迟导入模块属性

        Args:
            module_name: 模块名
            attr_name: 属性名
            fallback: 降级值

        Returns:
            延迟加载对象
        """
        key = (module_name, attr_name)
        if key not in self._attrs:
            self._attrs[key] = lazy_import(module_name, attr_name, fallback=fallback)
        return self._attrs[key]

    def is_available(self, module_name: str) -> bool:
        """
        检查模块是否可用

        Args:
            module_name: 模块名

        Returns:
            是否可用
        """
        if module_name in self._checked:
            return self._checked[module_name]

        try:
            importlib.import_module(module_name)
            self._checked[module_name] = True
            return True
        except ImportError:
            self._checked[module_name] = False
            return False

    def ensure_available(self, module_name: str, message: Optional[str] = None) -> None:
        """
        确保模块可用，否则抛出异常

        Args:
            module_name: 模块名
            message: 错误消息
        """
        if not self.is_available(module_name):
            msg = message or f"模块 {module_name} 不可用，请安装依赖"
            raise LazyImportError(msg)


def check_dependency(module_name: str) -> bool:
    """
    检查依赖是否可用

    Args:
        module_name: 模块名

    Returns:
        是否可用
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def require_dependency(module_name: str, message: Optional[str] = None) -> Callable:
    """
    装饰器：要求依赖可用

    Args:
        module_name: 模块名
        message: 错误消息

    Returns:
        装饰器

    Usage:
        @require_dependency('torch')
        def train_model():
            # 使用 torch
            pass
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if not check_dependency(module_name):
                msg = message or f"需要 {module_name} 依赖"
                raise LazyImportError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class LazyDependencies:
    """
    常用依赖的延迟加载封装

    Usage:
        from core.lazy_import import lazy_deps

        # 使用延迟加载的 torch
        if lazy_deps.torch:
            model = lazy_deps.torch.nn.Module()

        # 使用延迟加载的 numpy
        arr = lazy_deps.numpy.array([1, 2, 3])

        # 检查依赖是否可用
        has_torch = lazy_deps.has('torch')
        has_cv2 = lazy_deps.has('cv2')
    """

    def __init__(self):
        self._lazy = LazyModule()
        self._cache = {}

    def __getattr__(self, name: str):
        if name in self._cache:
            return self._cache[name]

        module_map = {
            'torch': 'torch',
            'numpy': 'numpy',
            'cv2': 'cv2',
            'PIL': 'PIL',
            'Image': ('PIL', 'Image'),
            'onnxruntime': 'onnxruntime',
            'faiss': 'faiss',
            'faiss_cpu': 'faiss_cpu',
            'easyocr': 'easyocr',
            'bcrypt': 'bcrypt',
            'redis': 'redis',
            'requests': 'requests',
            'pandas': 'pandas',
            'scipy': 'scipy',
            'sklearn': 'sklearn',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'tensorflow': 'tensorflow',
            'keras': 'keras',
            'transformers': 'transformers',
            'datasets': 'datasets',
            'accelerate': 'accelerate',
            'sqlalchemy': 'sqlalchemy',
            'psutil': 'psutil',
            'dotenv': 'dotenv',
        }

        if name in module_map:
            mapping = module_map[name]
            if isinstance(mapping, tuple):
                module_name, attr_name = mapping
                result = self._lazy.import_attr(module_name, attr_name, fallback=None)
            else:
                result = self._lazy.import_module(mapping, fallback=None)
            self._cache[name] = result
            return result

        raise AttributeError(f"'LazyDependencies' object has no attribute '{name}'")

    def has(self, name: str) -> bool:
        """
        检查依赖是否可用

        Args:
            name: 依赖名

        Returns:
            是否可用
        """
        return self._lazy.is_available(name)

    def ensure(self, name: str, message: Optional[str] = None) -> None:
        """
        确保依赖可用

        Args:
            name: 依赖名
            message: 错误消息
        """
        self._lazy.ensure_available(name, message)


lazy_deps = LazyDependencies()


class MemUsageReader:
    """
    内存使用读取器（延迟加载）
    内存使用检测设计
    """

    _reader = None
    _checked = False

    @classmethod
    def _get_reader(cls):
        if cls._checked:
            return cls._reader

        cls._checked = True
        try:
            import psutil
            process = psutil.Process()

            def reader():
                mem = process.memory_info()
                return mem.rss / 1024 ** 2, mem.vms / 1024 ** 2

            cls._reader = reader
        except Exception:
            cls._reader = None

        return cls._reader

    @classmethod
    def get_usage_mb(cls):
        """
        获取内存使用量（MB）

        Returns:
            (rss, vms) 或 (None, None)
        """
        reader = cls._get_reader()
        if reader is None:
            return None, None
        try:
            return reader()
        except Exception:
            return None, None

    @classmethod
    def get_rss_mb(cls):
        """
        获取 RSS 内存使用量（MB）

        Returns:
            RSS 内存使用量或 None
        """
        rss, _ = cls.get_usage_mb()
        return rss

    @classmethod
    def get_vms_mb(cls):
        """
        获取 VMS 内存使用量（MB）

        Returns:
            VMS 内存使用量或 None
        """
        _, vms = cls.get_usage_mb()
        return vms