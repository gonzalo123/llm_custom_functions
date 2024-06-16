import inspect

from langchain_core.tools import StructuredTool

from . import tools


def get_tool_functions(module):
    return {name: obj for name, obj in inspect.getmembers(module) if isinstance(obj, StructuredTool)}



def tool_decorator(func):
    func._is_tool = True
    return func


for name, obj in inspect.getmembers(tools):
    if inspect.isfunction(obj) and hasattr(obj, '_tool_metadata'):
        obj._is_tool = True

tools = get_tool_functions(tools)

__all__ = ['tools']
