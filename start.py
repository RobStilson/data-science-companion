"""Launcher that patches Python 3.14 + nest_asyncio + anyio incompatibility.

Root cause: Python 3.14 made asyncio.current_task() a C built-in that reads
C-level storage, but _PyTask.__step (used by nest_asyncio) calls the Python
_py_enter_task which only updates the Python _current_tasks dict. So
asyncio.current_task() returns None inside tasks, breaking sniffio and anyio.

Fix: replace asyncio.current_task with a Python function that reads from the
Python _current_tasks dict. Apply BEFORE importing anyio (which binds
current_task at import time).
"""
import asyncio
import asyncio.tasks
import asyncio.events

_orig_current_task = asyncio.current_task


def _patched_current_task(loop=None):
    if loop is None:
        try:
            loop = asyncio.events.get_running_loop()
        except RuntimeError:
            return None
    task = asyncio.tasks._current_tasks.get(loop)
    if task is not None:
        return task
    try:
        return _orig_current_task()
    except Exception:
        return None


asyncio.current_task = _patched_current_task
asyncio.tasks.current_task = _patched_current_task

import sys
import os

# truststore makes Python use the Windows certificate store, which includes
# certificates injected by security software (e.g. Avast Web/Mail Shield).
# Must be applied before any SSL connection is made.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    # Fallback: point at certifi's CA bundle
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass

if __name__ == "__main__":
    port = "8080"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--port", "-p") and i + 1 < len(args):
            port = args[i + 1]
            i += 2
        else:
            i += 1

    sys.argv = ["chainlit", "run", "app.py", "--port", port, "--headless"]

    from chainlit.cli import cli
    cli(standalone_mode=False)
