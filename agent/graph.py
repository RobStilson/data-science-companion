from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import classify, data_dict, descriptive, missing, skewness
from agent.state import AgentState


def build_graph():
    """Build and compile the auto-EDA LangGraph for the post-ingest sequence."""
    workflow = StateGraph(AgentState)

    workflow.add_node("missing", missing.run)
    workflow.add_node("classify", classify.run)
    workflow.add_node("descriptive", descriptive.run)
    workflow.add_node("skewness", skewness.run)
    workflow.add_node("data_dict", data_dict.run)

    workflow.add_edge(START, "missing")
    workflow.add_edge("missing", "classify")
    workflow.add_edge("classify", "descriptive")
    workflow.add_edge("descriptive", "skewness")
    workflow.add_edge("skewness", "data_dict")
    workflow.add_edge("data_dict", END)

    return workflow.compile()
