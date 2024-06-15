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
