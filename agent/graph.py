from langgraph.graph import StateGraph, END, START

from core.state import AgentState

from agent.security import gatekeeper_node
from agent.nodes import (
    retrieve_node,
    grade_documents_node,
    web_search_node,
    generate_answer_node,
    reject_node
)
from agent.edges import check_security_router, decide_to_generate

def create_agent_graph():
    """
    Assembles and compiles the LangGraph StateGraph for the Clinical Trial Navigator.
    """
    # 1. Initialize the Graph with our AgentState
    workflow = StateGraph(AgentState)

    # 2. Register all the Nodes
    workflow.add_node("gatekeeper", gatekeeper_node)
    workflow.add_node("rejection", reject_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_answer_node)

    # 3. Define the Orchestration Logic (Edges)

    # The journey begins at the Security Gatekeeper
    workflow.add_edge(START, "gatekeeper")

    # Security Routing: Decide if the query is safe and on-topic
    workflow.add_conditional_edges(
        "gatekeeper",
        check_security_router,
        {
            "safe": "retrieve",
            "unsafe": "rejection"
        }
    )

    # After retrieval, the state always moves to the grader
    workflow.add_edge("retrieve", "grade_documents")

    # Intelligence Routing: Decide if internal docs are enough or if we hit the web
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate"
        }
    )

    # If web search was used, move to final generation
    workflow.add_edge("web_search", "generate")

    # Both paths (Answer or Rejection) lead to the end of the process
    workflow.add_edge("generate", END)
    workflow.add_edge("rejection", END)

    # 4. Compile the Agent
    app = workflow.compile()
    return app

# Expose a compiled instance to be imported directly by the main app
agent_app = create_agent_graph()
