from __future__ import annotations

import chainlit as cl

from agent.nodes import ingest
from agent.state import AgentState, initial_state

_WELCOME = (
    "# Data Science Companion\n\n"
    "Upload a **CSV** or **Excel** (.xlsx) file to begin your analysis.\n"
    "I'll walk you through data quality, descriptive statistics, correlations, and more."
)

_ACCEPT = [
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
]


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
    await cl.Message(
        content="Please upload a CSV or Excel file to get started."
    ).send()


async def _handle_file(file: cl.File) -> None:
    state: AgentState = cl.user_session.get("state") or initial_state()
    state = await ingest.run(state, file.path, file.name)
    for msg in state["messages"]:
        await cl.Message(content=msg).send()
    cl.user_session.set("state", state)
