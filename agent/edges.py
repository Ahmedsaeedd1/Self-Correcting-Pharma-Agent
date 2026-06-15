from typing import Literal
from core.state import AgentState

def check_security_router(state: AgentState) -> Literal["safe", "unsafe"]:
    """
    Routes the graph path based on gatekeeper results.
    """
    if state["is_safe"] and state["is_in_domain"]:
        print("[LOG] Routing to retrieval pipeline.")
        return "safe"
    
    print("[LOG] Routing to rejection handler.")
    return "unsafe"

def decide_to_generate(state: AgentState) -> Literal["search", "generate"]:
    """
    Determines whether to proceed to generation or fallback to web search.
    """
    if state["search_needed"]:
        print("[LOG] Routing to Web Search.")
        return "search"
    
    print("[LOG] Routing to Generation.")
    return "generate"
