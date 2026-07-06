from __future__ import annotations

# tempfile lets us write the export scripts to a temporary location on disk
# so Chainlit can attach them as downloadable files.
import tempfile

import chainlit as cl
import plotly.graph_objects as go

from agent.graph import build_graph
from agent.nodes import correlations, export, ingest
from agent.nodes import visualize as viz_node
from agent.state import AgentState, initial_state

# Welcome message shown at the top of every new chat session.
_WELCOME = (
    "# Data Science Companion\n\n"
    "Upload a **CSV** or **Excel** (.xlsx) file to begin your analysis.\n\n"
    "After the auto-EDA I'll wait for your commands:\n"
    "- `correlate on <column>` — correlation analysis against a target column\n"
    "- `histogram of <col>`, `scatter of <col1> vs <col2>`, `box of <col>` — charts\n"
    "- `suggest charts` — AI chart recommendations\n"
    "- `export` — generate Python & R scripts"
)

# MIME types Chainlit will accept from the file picker.
_ACCEPT = [
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
]

# Build the LangGraph pipeline once at module load time (not per-request).
_GRAPH = build_graph()


# ── Chainlit lifecycle hooks ──────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start() -> None:
    # Called once when a user opens the chat. We store blank state in the
    # Chainlit user session so all subsequent handlers can read and update it.
    cl.user_session.set("state", initial_state())
    await cl.Message(content=_WELCOME).send()
    # AskFileMessage pauses execution until the user uploads a file.
    files = await cl.AskFileMessage(
        content="Please upload your data file to get started:",
        accept=_ACCEPT,
        max_size_mb=200,
    ).send()
    if files:
        await _handle_file(files[0])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    # Called every time the user sends a message or drops a file into the chat.
    for element in message.elements:
        if isinstance(element, cl.File):
            # If the message contains a file attachment, treat it as a new upload.
            await _handle_file(element)
            return

    state: AgentState = cl.user_session.get("state") or initial_state()
    if state["df"] is None:
        await cl.Message(content="Please upload a data file first.").send()
        return

    # Route the text message to the appropriate handler.
    await _dispatch(message.content.strip(), state)


# ── File handling ─────────────────────────────────────────────────────────────

async def _handle_file(file: cl.File) -> None:
    state: AgentState = cl.user_session.get("state") or initial_state()

    # Step 1: ingest — loads the file into state["df"] and sends a preview.
    # Called directly (not via the graph) because it needs the file path and name.
    state = await ingest.run(state, file.path, file.name)
    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    state = {**state, "messages": []}

    if state["df"] is None:
        # Ingest failed (bad format, oversized, etc.) — stop here.
        cl.user_session.set("state", state)
        return

    # Step 2: auto-EDA — run the full LangGraph pipeline.
    # ainvoke is the async version; it walks every node and returns the final state.
    await cl.Message(content="Running automated EDA…").send()
    prev_log_len = len(state["session_log"])
    state = await _GRAPH.ainvoke(state)
    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    state = {**state, "messages": []}

    # Step 3: display any charts the auto_viz node generated.
    # We check only entries added by this graph run (slice from prev_log_len onward).
    for entry in state["session_log"][prev_log_len:]:
        if "figure" in entry:
            # Reconstruct a Plotly Figure from the dict stored in the session log.
            fig = go.Figure(entry["figure"])
            label = ", ".join(entry["cols"]) if entry["cols"] else "all numeric"
            await cl.Message(
                content="",
                elements=[cl.Plotly(figure=fig, display="inline", name=label)],
            ).send()

    cl.user_session.set("state", state)
    await _send_next_step_actions(state)


async def _send_next_step_actions(state: AgentState) -> None:
    # Offer clickable action buttons so users can explore next steps without
    # having to remember the exact command syntax.
    actions = [
        cl.Action(name="next_step", label="Suggest Charts", payload={"cmd": "suggest charts"}),
    ]
    if len(state["num_cols"]) >= 2:
        # Only offer heatmap if there are at least 2 numeric columns to compare.
        actions.append(
            cl.Action(name="next_step", label="Correlation Heatmap", payload={"cmd": "heatmap"})
        )
    # Pre-fill a "Correlate on X" button using the last continuous column as a
    # reasonable default outcome variable.
    target = (state["continuous_cols"] or state["num_cols"] or [None])[-1]
    if target:
        actions.append(
            cl.Action(
                name="next_step",
                label=f"Correlate on '{target}'",
                payload={"cmd": f"correlate on {target}"},
            )
        )
    actions.append(
        cl.Action(name="next_step", label="Export Scripts", payload={"cmd": "export"})
    )
    await cl.Message(
        content="Auto-EDA complete. What would you like to explore next?",
        actions=actions,
    ).send()


@cl.action_callback("next_step")
async def on_next_step(action: cl.Action) -> None:
    # Chainlit calls this when the user clicks one of the action buttons.
    # We extract the pre-built command string from the button's payload and
    # run it through the same _dispatch function that handles typed commands.
    state: AgentState = cl.user_session.get("state") or initial_state()
    cmd = (action.payload or {}).get("cmd", "")
    if cmd:
        await _dispatch(cmd, state)


# ── Command dispatcher ────────────────────────────────────────────────────────

async def _dispatch(text: str, state: AgentState) -> None:
    # Routes a text command to the right node and then sends results to the UI.
    lower = text.lower()
    new_figure: go.Figure | None = None  # set if a chart was produced
    _is_export = False                   # set if the export node ran

    if lower.startswith("correlate on "):
        # Extract the column name after "correlate on " and store it in state
        # so correlations.run() knows which column to treat as the outcome.
        col = text[len("correlate on "):].strip()
        state = {**state, "outcome_col": col}
        state = await correlations.run(state)

    elif lower.startswith("suggest chart"):
        # Ask the LLM for chart ideas based on the dataset's column profile.
        state = await viz_node.suggest(state)
        suggestions = state["viz_suggestions"]
        msg = "**Suggested charts:**\n" + "\n".join(f"- {s}" for s in suggestions)
        await cl.Message(content=msg).send()
        state = {**state, "messages": []}

    elif _is_viz_command(lower):
        chart_type, cols = _parse_viz(text)
        if chart_type:
            prev_log_len = len(state["session_log"])
            state = await viz_node.render(state, chart_type, cols)
            # Pull the figure out of the session log so we can send it to the UI below.
            for entry in state["session_log"][prev_log_len:]:
                if "figure" in entry:
                    new_figure = go.Figure(entry["figure"])
                    break
        else:
            await cl.Message(
                content="Couldn't parse chart command. Try: `histogram age` or `scatter age vs salary`"
            ).send()
            return

    elif lower in ("export", "generate code", "export code"):
        state = await export.run(state)
        _is_export = True

    else:
        await cl.Message(
            content=(
                "I didn't recognise that command.\n\n"
                "Try: `correlate on <col>`, `histogram <col>`, "
                "`scatter <col1> and <col2>`, `suggest charts`, or `export`."
            )
        ).send()
        return

    # Send any text messages the node appended, then clear the queue.
    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    state = {**state, "messages": []}

    # Send the chart inline if one was produced.
    if new_figure is not None:
        await cl.Message(
            content="",
            elements=[cl.Plotly(figure=new_figure, display="inline", name="chart")],
        ).send()

    # For export commands, also attach the scripts as downloadable files.
    if _is_export:
        # Find the most recent export entry in the session log.
        export_entry = next(
            (e for e in reversed(state["session_log"]) if e.get("step") == "export"),
            None,
        )
        if export_entry:
            # Strip the file extension to use as the stem of the download filename.
            stem = (state.get("filename") or "analysis").rsplit(".", 1)[0]
            elements = []
            for code, suffix, label, mime in [
                (export_entry["python"], ".py", f"{stem}.py", "text/x-python"),
                (export_entry["r"], ".R", f"{stem}.R", "text/x-r"),
            ]:
                # Write to a temp file so Chainlit can serve it as an attachment.
                # delete=False keeps the file alive after .close() so Chainlit can read it.
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=suffix, delete=False, encoding="utf-8"
                )
                tmp.write(code)
                tmp.close()
                elements.append(cl.File(name=label, path=tmp.name, display="inline", mime=mime))
            await cl.Message(content="Download your scripts:", elements=elements).send()

    cl.user_session.set("state", state)


# ── Chart command parsing ─────────────────────────────────────────────────────

# Ordered list of (keyword, chart_type) pairs. "pair plot" must come first
# because it contains a space — checking it before single-word keywords avoids
# "pair plot" being partially matched by "pair" (which doesn't exist here but
# illustrates why order matters when keywords are substrings of each other).
_CHART_KEYWORDS = [
    ("pair plot", "pair plot"),
    ("histogram", "histogram"),
    ("scatter", "scatter"),
    ("heatmap", "heatmap"),
    ("line", "line"),
    ("box", "box"),
    ("bar", "bar"),
]


def _is_viz_command(lower: str) -> bool:
    # "heatmap" alone (no columns) is a valid command.
    if lower.strip() == "heatmap":
        return True
    return any(lower.startswith(kw + " ") for kw, _ in _CHART_KEYWORDS)


def _parse_viz(text: str) -> tuple[str | None, list[str]]:
    # Converts a natural-language chart command into (chart_type, [col_names]).
    # Handles many phrasings: "histogram age", "histogram of age",
    # "scatter age vs salary", "scatter age and salary", "scatter chart of age vs salary"
    if text.strip().lower() == "heatmap":
        return "heatmap", []
    lower = text.lower()
    for keyword, chart_type in _CHART_KEYWORDS:
        if lower.startswith(keyword + " "):
            rest = text[len(keyword):].lstrip()
            # Strip optional filler words users naturally add ("of", "chart", "plot").
            for filler in ("chart of ", "plot of ", "chart ", "plot ", "of "):
                if rest.lower().startswith(filler):
                    rest = rest[len(filler):]
                    break
            rest = rest.strip()
            lower_rest = rest.lower()
            # Support both "vs" and "and" as two-column separators.
            if " vs " in lower_rest:
                split_at = lower_rest.index(" vs ")
                cols = [rest[:split_at].strip(), rest[split_at + 4:].strip()]
            elif " and " in lower_rest:
                split_at = lower_rest.index(" and ")
                cols = [rest[:split_at].strip(), rest[split_at + 5:].strip()]
            else:
                cols = [rest.strip()]
            # Filter out empty strings that can appear if the user typed trailing spaces.
            return chart_type, [c for c in cols if c]
    return None, []
