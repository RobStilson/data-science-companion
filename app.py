from __future__ import annotations

import tempfile

import chainlit as cl
import plotly.graph_objects as go

from agent.graph import build_graph
from agent.nodes import correlations, export, ingest
from agent.nodes import visualize as viz_node
from agent.state import AgentState, initial_state

_WELCOME = (
    "# Data Science Companion\n\n"
    "Upload a **CSV** or **Excel** (.xlsx) file to begin your analysis.\n\n"
    "After the auto-EDA I'll wait for your commands:\n"
    "- `correlate on <column>` — correlation analysis against a target column\n"
    "- `histogram of <col>`, `scatter of <col1> vs <col2>`, `box of <col>` — charts\n"
    "- `suggest charts` — AI chart recommendations\n"
    "- `export` — generate Python & R scripts"
)

_ACCEPT = [
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
]

_GRAPH = build_graph()


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("state", initial_state())
    await cl.Message(content=_WELCOME).send()
    files = await cl.AskFileMessage(
        content="Please upload your data file to get started:",
        accept=_ACCEPT,
        max_size_mb=200,
    ).send()
    if files:
        await _handle_file(files[0])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    for element in message.elements:
        if isinstance(element, cl.File):
            await _handle_file(element)
            return

    state: AgentState = cl.user_session.get("state") or initial_state()
    if state["df"] is None:
        await cl.Message(content="Please upload a data file first.").send()
        return

    await _dispatch(message.content.strip(), state)


async def _handle_file(file: cl.File) -> None:
    state: AgentState = cl.user_session.get("state") or initial_state()

    # Ingest (needs file path + name — called directly, not through graph)
    state = await ingest.run(state, file.path, file.name)
    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    state = {**state, "messages": []}

    if state["df"] is None:
        cl.user_session.set("state", state)
        return

    # Auto-EDA sequence via LangGraph
    await cl.Message(content="Running automated EDA…").send()
    prev_log_len = len(state["session_log"])
    state = await _GRAPH.ainvoke(state)
    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    state = {**state, "messages": []}

    # Emit auto-generated charts produced by the auto_viz node
    for entry in state["session_log"][prev_log_len:]:
        if "figure" in entry:
            fig = go.Figure(entry["figure"])
            label = ", ".join(entry["cols"]) if entry["cols"] else "all numeric"
            await cl.Message(
                content="",
                elements=[cl.Plotly(figure=fig, display="inline", name=label)],
            ).send()

    cl.user_session.set("state", state)
    await _send_next_step_actions(state)


async def _send_next_step_actions(state: AgentState) -> None:
    actions = [
        cl.Action(name="next_step", label="Suggest Charts", payload={"cmd": "suggest charts"}),
    ]
    if len(state["num_cols"]) >= 2:
        actions.append(
            cl.Action(name="next_step", label="Correlation Heatmap", payload={"cmd": "heatmap"})
        )
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
    state: AgentState = cl.user_session.get("state") or initial_state()
    cmd = (action.payload or {}).get("cmd", "")
    if cmd:
        await _dispatch(cmd, state)


async def _dispatch(text: str, state: AgentState) -> None:
    lower = text.lower()
    new_figure: go.Figure | None = None
    _is_export = False

    if lower.startswith("correlate on "):
        col = text[len("correlate on "):].strip()
        state = {**state, "outcome_col": col}
        state = await correlations.run(state)

    elif lower.startswith("suggest chart"):
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

    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    state = {**state, "messages": []}

    if new_figure is not None:
        await cl.Message(
            content="",
            elements=[cl.Plotly(figure=new_figure, display="inline", name="chart")],
        ).send()

    if _is_export:
        export_entry = next(
            (e for e in reversed(state["session_log"]) if e.get("step") == "export"),
            None,
        )
        if export_entry:
            stem = (state.get("filename") or "analysis").rsplit(".", 1)[0]
            elements = []
            for code, suffix, label, mime in [
                (export_entry["python"], ".py", f"{stem}.py", "text/x-python"),
                (export_entry["r"], ".R", f"{stem}.R", "text/x-r"),
            ]:
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=suffix, delete=False, encoding="utf-8"
                )
                tmp.write(code)
                tmp.close()
                elements.append(cl.File(name=label, path=tmp.name, display="inline", mime=mime))
            await cl.Message(content="Download your scripts:", elements=elements).send()

    cl.user_session.set("state", state)


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
    if lower.strip() == "heatmap":
        return True
    return any(lower.startswith(kw + " ") for kw, _ in _CHART_KEYWORDS)


def _parse_viz(text: str) -> tuple[str | None, list[str]]:
    if text.strip().lower() == "heatmap":
        return "heatmap", []
    lower = text.lower()
    for keyword, chart_type in _CHART_KEYWORDS:
        if lower.startswith(keyword + " "):
            rest = text[len(keyword):].lstrip()
            # strip optional filler words users naturally add
            for filler in ("chart of ", "plot of ", "chart ", "plot ", "of "):
                if rest.lower().startswith(filler):
                    rest = rest[len(filler):]
                    break
            rest = rest.strip()
            lower_rest = rest.lower()
            if " vs " in lower_rest:
                split_at = lower_rest.index(" vs ")
                cols = [rest[:split_at].strip(), rest[split_at + 4:].strip()]
            elif " and " in lower_rest:
                split_at = lower_rest.index(" and ")
                cols = [rest[:split_at].strip(), rest[split_at + 5:].strip()]
            else:
                cols = [rest.strip()]
            return chart_type, [c for c in cols if c]
    return None, []
