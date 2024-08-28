from __future__ import annotations

from functools import partial, wraps
from typing import TYPE_CHECKING

from batched.utils import is_method

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Optional

    from batched.types import BatchFunc


def _dynamic_batch(
    make_processor: Callable,
    func: Optional[BatchFunc] = None,
):
    class _InstanceBatchProcessor:
        def __init__(self, _func: BatchFunc):
            self.func = _func

        def __get__(self, instance, owner):
            func_name = f"__batched_{self.func.__name__}"
            if func_name not in instance.__dict__:
                _func = partial(self.func, instance)
                processor = make_processor(_func)
                instance.__dict__[func_name] = processor

            return instance.__dict__[func_name]

    def _batch_decorator(_func: BatchFunc):
        batch_func = _InstanceBatchProcessor(_func) if is_method(_func) else make_processor(_func)

        return wraps(_func)(batch_func)

    if func is None:
        return _batch_decorator
    return _batch_decorator(func)
