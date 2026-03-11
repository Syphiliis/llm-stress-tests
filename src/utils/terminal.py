"""
Terminal helpers with a graceful fallback when colorama is unavailable.
"""

try:
    from colorama import Fore as _Fore
    from colorama import Style as _Style
    from colorama import init as _colorama_init
except ImportError:
    class _NoColor:
        def __getattr__(self, _: str) -> str:
            return ""

    Fore = _NoColor()
    Style = _NoColor()

    def init(*_: object, **__: object) -> None:
        return None
else:
    Fore = _Fore
    Style = _Style

    def init(*args: object, **kwargs: object) -> None:
        _colorama_init(*args, **kwargs)
