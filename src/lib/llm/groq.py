import logging

from langchain_groq import ChatGroq

from settings import MODEL, GROQ_API_KEY

logger = logging.getLogger(__name__)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=MODEL,
)

logger.info(f"Groq Model {MODEL} loaded")
