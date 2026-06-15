from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the internal state passed between all nodes in the LangGraph application.
    
    Attributes:
        messages: The chat history and standard AI/Human messages.
        is_safe: Result from the gatekeeper node indicating if the input is benign.
        is_in_domain: Result from the gatekeeper node indicating if the query is in the medical domain.
        documents: Raw text chunks extracted from Pinecone or Web Search.
        search_needed: Flag set by the grader node if internal documents aren't sufficient.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    is_safe: bool
    is_in_domain: bool
    documents: List[str]
    search_needed: bool
