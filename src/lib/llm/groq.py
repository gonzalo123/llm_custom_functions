import logging

from langchain_groq import ChatGroq

from settings import OLLAMA_MODEL, GROQ_API_KEY

logger = logging.getLogger(__name__)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=OLLAMA_MODEL,
)

logger.info(f"Groq Model {OLLAMA_MODEL} loaded")
