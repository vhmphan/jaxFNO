"""Temporary plotting caches, removed when plotting finishes."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def _clear_cached_paths():
    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is not None:
        for name in ("get_cachedir", "get_configdir"):
            clear = getattr(getattr(matplotlib, name, None), "cache_clear", None)
            if clear is not None:
                clear()


@contextmanager
def temporary_plot_cache():
    """Keep font/config caches outside results and clean up even on failure."""
    names = ("MPLCONFIGDIR", "XDG_CACHE_HOME")
    previous = {name: os.environ.get(name) for name in names}
    with TemporaryDirectory(prefix="jaxfno-plot-") as directory:
        try:
            os.environ["MPLCONFIGDIR"] = str(Path(directory) / "matplotlib")
            os.environ["XDG_CACHE_HOME"] = str(Path(directory) / "cache")
            _clear_cached_paths()
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _clear_cached_paths()
