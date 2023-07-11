from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def nest(fun: Callable[[T], T], n: int) -> Callable[[T], T]:
    """Helper function to apply a function onto itself n times."""

    def nested_fun(x: T) -> T:
        for _ in range(n):
            x = fun(x)

        return x

    return nested_fun


class defaultdict_keyed(defaultdict[_KT, _VT]):
    """A default dictionary where the factory depends on the missing key"""

    default_factory: Callable[[_KT], _VT]

    def __init__(self, __factory: Callable[[_KT], _VT]):
        super().__init__()
        self.default_factory = __factory

    def __missing__(self, __key: _KT) -> _VT:
        if not self.default_factory:
            return super().__missing__(__key)

        value = self.default_factory(__key)
        self[__key] = value
        return value
