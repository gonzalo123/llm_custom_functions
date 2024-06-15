import logging

from lib.chains.math_chain.chain import get_chain
from lib.llm.groq import llm

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level='INFO',
    datefmt='%d/%m/%Y %X')

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    chain = get_chain(llm)

    user_prompts = [
        "How much is five times twelve?",
        "How much is five plus twelve?",
        "How much is twelve minus five?",
    ]

    for prompt in user_prompts:
        responses = chain.ask_question(prompt)
        for response in responses:
            print(f"Q: {prompt} R:{response}")
