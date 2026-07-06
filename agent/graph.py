from __future__ import annotations

# LangGraph models an agent as a directed graph where each node is a function
# and edges define the order of execution. Think of it like a pipeline where
# state flows from node to node, each one reading and updating the same dict.
from langgraph.graph import END, START, StateGraph

from agent.nodes import auto_viz, classify, data_dict, descriptive, missing, skewness
from agent.state import AgentState

# Module-level cache so we only compile the graph once per process.
# Compiling resolves all the edges into an executable plan — it's cheap
# but no reason to repeat it on every file upload.
_compiled = None


def build_graph():
    """Build and compile the auto-EDA LangGraph for the post-ingest sequence."""
    global _compiled
    if _compiled is not None:
        return _compiled

    # StateGraph takes the TypedDict class so LangGraph knows the shape of state.
    workflow = StateGraph(AgentState)

    # Register each analysis step as a named node.
    # The string name is what we reference when adding edges below.
    workflow.add_node("missing", missing.run)
    workflow.add_node("classify", classify.run)
    workflow.add_node("descriptive", descriptive.run)
    workflow.add_node("skewness", skewness.run)
    workflow.add_node("data_dict", data_dict.run)
    workflow.add_node("auto_viz", auto_viz.run)

    # Define the linear execution order. START and END are LangGraph sentinels.
    # classify runs before descriptive/auto_viz because those nodes need
    # num_cols and cat_cols to already be populated in state.
    workflow.add_edge(START, "missing")
    workflow.add_edge("missing", "classify")
    workflow.add_edge("classify", "descriptive")
    workflow.add_edge("descriptive", "skewness")
    workflow.add_edge("skewness", "data_dict")
    workflow.add_edge("data_dict", "auto_viz")
    workflow.add_edge("auto_viz", END)

    # compile() validates the graph (no orphan nodes, no unintended cycles)
    # and returns an object with .invoke() / .ainvoke() methods.
    _compiled = workflow.compile()
    return _compiled
