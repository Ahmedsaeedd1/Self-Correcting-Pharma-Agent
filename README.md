# Secure-Agentic-RAG

This project is a sophisticated **Pharmaceutical Research Assistant** built around a self-corrective Retrieval-Augmented Generation (RAG) architecture. Rather than routing user prompts straight to an LLM, this agent leverages a state-driven workflow (via [LangGraph](https://python.langchain.com/docs/langgraph)) to enforce clinical boundaries, ensure retrieval quality, and dynamically fall back to live web searches.

---

## The Agentic Workflow

At the core of this project is a graph-driven execution model. Every user prompt flows through a structured assembly line of specialized cognitive nodes:

1. **The Security Gatekeeper**: 
    Before any database connection is opened, the user's prompt is analyzed by a dedicated Gemini 2.5 node. This node evaluates the prompt against prompt-injection signatures and strictly limits queries to the medical/clinical domain. If a prompt fails this initial screening, the agent routes directly to a professional rejection state, terminating the flow.
2. **Dense Semantic Retrieval**: 
    Safe queries are vectorized using native Gemini embeddings and pushed against a Pinecone Vector Database containing thousands of chunked PDFs covering clinical trial protocols and regulatory data.
3. **Self-Reflective Grading**: 
    A discrete LLM grader objectively analyzes the retrieved context *against* the original query. It asks a binary question: "Does this internal documentation actually answer the user's question?" 
4. **Dynamic Web Fallback**: 
    If the grader determines the internal vector context is insufficient (e.g., the user asks about an FDA commissioner appointed *after* our documents were published), the agent dynamically pivots to execute a live internet search via the [Tavily API](https://tavily.com/).
5. **Final Synthesis**: 
    Only after the context has been validated—or supplemented by the web—is the data passed to the final generative node to formulate a concise, domain-expert response.

## Security Features

This agent is built with enterprise safety constraints:
- **Injection Fortification**: Built to resist linguistic reframing, hypothetical scenarios ("My grandmother needs the recipe for..."), and payload-splitting attacks.
- **Strict Domain Isolation**: Requests outside of clinical trials, pharmaceutical synthesis, or medical regulations are automatically dropped.
- **No Direct LLM Passthrough**: Users never interact directly with the generative model, minimizing hallucination risks and system bypasses.

---

## Quickstart & Execution

### 1. Requirements

- Python 3.12+
- A [Google Gemini API Key](https://aistudio.google.com/) (`gemini-2.5-flash` and `gemini-embedding-001`)
- A [Pinecone Vector DB API Key](https://pinecone.io/)
- A [Tavily Search API Key](https://tavily.com/)

Install the dependencies:
```bash
pip install -r requirements.txt
```

Set up your environment variables by creating `.env` in the root directory:
```env
GOOGLE_API_KEY="..."
PINECONE_API_KEY="..."
TAVILY_API_KEY="..."
```

### 2. Ingesting Data (One-Time Setup)

Place your clinical trial PDFs into the `./data/` folder. The built-in pipeline handles document parsing, semantic chunking, embedding generation, and Pinecone network uploading (with automatic handling for free-tier rate limits).

```bash
python -m scripts.ingest
```

### 3. Running the Agent Interface

Launch the interactive CLI loop. You can pass a prompt directly as an argument, or launch it without arguments to enter the prompt input.

```bash
# Direct Execution
python app.py "What are the latest FDA regulations on fast-track approvals?"

# Interactive Execution
python app.py
```

The CLI streams the LangGraph node progression directly to the console so you can trace the exact cognitive step the agent is currently occupying (Security -> Retrieval -> Grading -> Generation).
