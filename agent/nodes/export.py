from __future__ import annotations

from agent.state import AgentState
from utils.code_gen import generate_python, generate_r


async def run(state: AgentState) -> AgentState:
    """Generate Python and R export scripts from the session log."""
    session_log = state["session_log"]
    filename = state["filename"]

    # Delegate to code_gen.py which knows how to translate each session_log
    # entry (missing data, correlations, charts, etc.) into runnable code.
    code_py = generate_python(session_log, filename)
    code_r = generate_r(session_log, filename)

    # Show the scripts in the chat as syntax-highlighted code blocks.
    # The actual downloadable files are written to temp files in app.py.
    msg_py = f"### Python Export\n```python\n{code_py}\n```"
    msg_r = f"### R Export\n```r\n{code_r}\n```"

    # Store the raw code strings in the session log so app.py can write
    # them to temporary files and attach them as downloadable attachments.
    log_entry = {"step": "export", "python": code_py, "r": code_r}

    return {
        **state,
        "messages": state["messages"] + [msg_py, msg_r],
        "session_log": state["session_log"] + [log_entry],
    }
