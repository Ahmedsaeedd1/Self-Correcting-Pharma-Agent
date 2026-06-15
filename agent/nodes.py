import os
import time
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from core.state import AgentState
from core.llm import llm
from core.config import INDEX_NAME
from ingestion.embedder import GeminiEmbeddings

# ---------------------------------------------------------------------------
# Quota Helper
# ---------------------------------------------------------------------------

def quota_wait(seconds: int = 60):
    """
    Mandatory pause to respect Gemini free-tier rate limits.
    """
    print(f"[LOG] System pause: {seconds} seconds to refresh API quota.")
    time.sleep(seconds)

# ---------------------------------------------------------------------------
# Tool Integrations
# ---------------------------------------------------------------------------

tavily_search = TavilySearchResults(max_results=3)

class RelevanceGrade(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

grader_llm = llm.with_structured_output(RelevanceGrade)

# Setup connection to Pinecone Vector DB
# Note: Doing this at the module level means it connects when the app starts
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
index = pc.Index(INDEX_NAME)
embeddings = GeminiEmbeddings()
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# ---------------------------------------------------------------------------
# Agent Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState):
    """
    Retrieves pharmaceutical documents from the Pinecone vector store.
    """
    query = state["messages"][-1].content
    print(f"[LOG] Initiating retrieval from Pinecone for query: {query[:50]}...")
    
    documents = vectorstore.similarity_search(query, k=3)
    doc_texts = [d.page_content for d in documents]
    
    print(f"[STATUS] Retrieval complete. Fetched {len(doc_texts)} chunks.")
    return {"documents": doc_texts}


def grade_documents_node(state: AgentState):
    """
    Determines if the retrieved documents are relevant to the user's query.
    """
    print("[LOG] Grading document relevance against user query.")
    quota_wait(60)
    
    query = state["messages"][-1].content
    docs = state["documents"]
    
    combined_docs = "\n\n".join(docs)
    
    system_prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question. "
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. "
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )
    
    result = grader_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User question: {query}\n\nRetrieved documents: {combined_docs}"}
    ])
    
    search_needed = (result.binary_score.lower() == "no")
    print(f"[STATUS] Relevance grade: {result.binary_score}. Web search required: {search_needed}")
    
    return {"search_needed": search_needed}


def web_search_node(state: AgentState):
    """
    Fallback node: Performs a web search if the internal database is insufficient.
    """
    query = state["messages"][-1].content
    print(f"[LOG] Local documents insufficient. Triggering Tavily web search for: {query[:50]}...")
    
    search_results = tavily_search.invoke({"query": query})
    
    # Format the search results into a string for the final generation
    search_content = "\n".join([r["content"] for r in search_results])
    
    print("[STATUS] Web search successful. Data appended to context.")
    return {"documents": [search_content]}


def generate_answer_node(state: AgentState):
    """
    Synthesizes the final answer using the filtered context.
    """
    print("[LOG] Generating final synthesis.")
    quota_wait(60)
    
    query = state["messages"][-1].content
    context = "\n\n".join(state["documents"])
    
    system_prompt = (
        "You are an expert Clinical Trial Navigator. Use the provided context to answer the question. "
        "If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
    )
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
    ])
    
    print("[STATUS] Final generation complete.")
    return {"messages": [AIMessage(content=response.content)]}


def reject_node(state: AgentState):
    """
    Provides a professional refusal for unsafe or off-domain queries.
    """
    print("[LOG] Query rejected by security gatekeeper.")
    return {
        "messages": [AIMessage(content="I am a specialized Pharmaceutical Research Assistant. I can only assist with clinical trials, drug regulations, and pharmaceutical data. Please refine your query.")]
    }
