from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import AGENT_MODEL

# We centralize the LLM instance here so we don't have to keep re-instantiating it
# with the 'gemini-2.5-flash' model throughout our different agent nodes.
# Temperature is set to 0 to keep the responses deterministic and clinical.
llm = ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=0)
