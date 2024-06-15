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
        try:
            user_message = HumanMessage(content=user_prompt)
            messages = [self.system_message, user_message]
            ai_msg = self.llm_with_tools.invoke(messages)

            for tool_call in ai_msg.tool_calls:
                tool_output = self.tools[tool_call["name"]].invoke(tool_call["args"])
                logger.info(f"Tool: '{tool_call['name']}' called output: {tool_output}")

            return ai_msg.content
        except Exception as e:
            logger.error(f"Error during question processing: {e}")
