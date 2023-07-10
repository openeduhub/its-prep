from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def nest(fun: Callable[[T], T], n: int) -> Callable[[T], T]:
    """Helper function to apply a function onto itself n times."""

    def nested_fun(x: T) -> T:
        for _ in range(n):
            x = fun(x)

        return x

    return nested_fun
