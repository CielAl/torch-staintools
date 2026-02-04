import torch
import functools
import warnings
from typing import Callable, Any, Optional, Protocol, cast
from torch_staintools.constants import CONFIG

_FIELD_COMPILED_ATTR = 'compiled_fn'


class CompiledWrapper(Protocol):
    compiled_fn: Optional[Callable]

    def reset_cache(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


def lazy_compile(func: Optional[Callable] = None,
                 *,
                 dynamic: Optional[bool] = None,
                 fullgraph: bool = False,
                 backend: Optional[str] = None,) -> CompiledWrapper | Callable[[Callable], CompiledWrapper]:
    """Enable or disable torch.compile by torch_staintools.constants.CONFIG.ENABLE_COMPILE.

    If True, function will be compiled and cached. Otherwise, it will be executed in eager mode.

    Args:
        func: The function to compile.

    Returns:
        CompiledWrapper: The compiled function or the original function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        enable_compile = getattr(CONFIG, "ENABLE_COMPILE", False)
        if not enable_compile:
            # if disabled execute it in eager mode
            return func(*args, **kwargs)

        if not hasattr(wrapper, _FIELD_COMPILED_ATTR) or wrapper.compiled_fn is None:
            try:
                wrapper.compiled_fn = torch.compile(func, )
            except Exception as e:
                warnings.warn(f"torch.compile failed for '{func.__name__}': {e}. "
                              f"Falling back to eager execution.")
                wrapper.compiled_fn = func

        return wrapper.compiled_fn(*args, **kwargs)

    wrapper = cast(CompiledWrapper, wrapper)
    # init the attribute
    wrapper.compiled_fn = None
    # clear the compiled cache --> future use
    def reset_cache():
        wrapper.compiled_fn = None
    wrapper.reset_cache = reset_cache
    return wrapper


def static_compile(func: Callable) -> Callable:
    """Import-time wrapper.

    CONFIG.ENABLE_COMPILE must be modified before importing any compiled functions.

    Args:
        func: The function to compile.

    Returns:
        The compiled function or the original function.
    """

    if getattr(CONFIG, "ENABLE_COMPILE", False):
        try:
            return torch.compile(func)
        except Exception as e:
            warnings.warn(f"torch.compile failed for '{func.__name__}': {e}. "
                          f"Falling back to eager execution.")
            return func

    return func