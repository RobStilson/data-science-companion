from __future__ import annotations

import chainlit as cl

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
    state = await _GRAPH.ainvoke(state)
    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    state = {**state, "messages": []}

    cl.user_session.set("state", state)
    await cl.Message(
        content="Auto-EDA complete. Type a command or ask a question."
    ).send()


async def _dispatch(text: str, state: AgentState) -> None:
    lower = text.lower()

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
            state = await viz_node.render(state, chart_type, cols)
        else:
            await cl.Message(
                content="Couldn't parse chart command. Try: `histogram of age`"
            ).send()
            return

    elif lower in ("export", "generate code", "export code"):
        state = await export.run(state)

    else:
        await cl.Message(
            content=(
                "I didn't recognise that command.\n\n"
                "Try: `correlate on <col>`, `histogram of <col>`, "
                "`scatter of <col1> vs <col2>`, `suggest charts`, or `export`."
            )
        ).send()
        return

    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    state = {**state, "messages": []}
    cl.user_session.set("state", state)


def _is_viz_command(lower: str) -> bool:
    return any(lower.startswith(p) for p in (
        "histogram of ", "box of ", "scatter of ",
        "bar of ", "line of ", "heatmap of ", "pair plot of ",
    ))


def _parse_viz(text: str) -> tuple[str | None, list[str]]:
    lower = text.lower()
    prefixes = [
        ("histogram of ", "histogram"),
        ("box of ", "box"),
        ("scatter of ", "scatter"),
        ("bar of ", "bar"),
        ("line of ", "line"),
        ("heatmap of ", "heatmap"),
        ("pair plot of ", "pair plot"),
    ]
    for prefix, chart_type in prefixes:
        if lower.startswith(prefix):
            rest = text[len(prefix):]  # preserve original case for column names
            lower_rest = rest.lower()
            if " vs " in lower_rest:
                split_at = lower_rest.index(" vs ")
                cols = [rest[:split_at].strip(), rest[split_at + 4:].strip()]
            else:
                cols = [rest.strip()]
            return chart_type, cols
    return None, []
