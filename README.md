# Integrating AI Models with Function Calls using Python and LangChain

Today we'll explore the integration of AI models with function calls using Python and LangChain. This example showcases how to leverage [LangChain](https://www.langchain.com/) for orchestrating AI and natural language processing tasks. In this example we´ll integrate AI models seamlessly with custom functions. While the functions used here are straightforward examples, such as basic arithmetic operations, they illustrate the foundational concepts applicable to more complex scenarios, such as invoking external APIs or more complicated processing pipelines. We need a LLM model with function calling capabilities (not all models allows us to call custom functions). For this example we're going to use Groq llm which has a public api (free) with function calling support. So we need to obtain an api key [here](https://console.groq.com/).

That's the main script. It only obtains the chain with our llm instance

```python
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
            print(f"Q: {prompt} A:{response}")
```

That's the chain

```python
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from .tools import tools

logger = logging.getLogger(__name__)


def get_chain(llm):
    return CustomMathChain(llm, tools)


class CustomMathChain:
    system_prompt_content = """
        You are a model that has various mathematical functions.
        You can only respond to questions related to functions that you know.
        """

    def __init__(self, llm, tools):
        self.llm_with_tools = llm.bind_tools(list(tools.values()))
        self.system_message = SystemMessage(content=self.system_prompt_content)
        self.tools = tools

    def ask_question(self, user_prompt):
        responses = []
        try:
            user_message = HumanMessage(content=user_prompt)
            messages = [self.system_message, user_message]
            ai_msg = self.llm_with_tools.invoke(messages)

            for tool_call in ai_msg.tool_calls:
                tool_output = self.tools[tool_call["name"]].invoke(tool_call["args"])
                logger.info(f"Tool: '{tool_call['name']}' called output: {tool_output}")
                responses.append(tool_output)

            return responses
        except Exception as e:
            logger.error(f"Error during question processing: {e}")
```

This custom chain utilizes functions defined here, employing the @tool decorator. It is crucial to properly define input and output variables and provide thorough documentation for our tools. AI leverages this information to determine the appropriate function call for each scenario. Various methods exist for defining our tools; here, I've opted for the simplest approach. For more detailed guidance on defining custom functions, refer to [this resource](https://python.langchain.com/v0.1/docs/modules/tools/toolkits/).

```python
from langchain_core.tools import tool


@tool
def ia_sum(a: int, b: int) -> int:
    """ Return the sum of `a` and `b` """
    return a + b


@tool
def ia_diff(a: int, b: int) -> int:
    """ Return the difference of `a` and `b` """
    return a - b


@tool
def ia_multiply(a: int, b: int) -> int:
    """ Return the product of `a` and `b` """
    return a * b


tools = {
    "ia_sum": ia_sum,
    "ia_diff": ia_diff,
    "ia_multiply": ia_multiply
}
```

And that’s all! Working with our custom functions is quite straightforward. As mentioned earlier, we’re using very simple functions (add, diff, and multiply). In reality, we don’t need an LLM or AI to perform these arithmetic operations. However, imagine integrating real-world functions that access APIs and your business models. AI can handle natural language processing to interpret user input and identify the correct function to execute the task.