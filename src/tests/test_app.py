from unittest.mock import MagicMock

from lib.chains.math_chain.chain import CustomMathChain


def test_multiply():
    mock_tools = MagicMock()
    mock_llm = MagicMock()
    mock_tools["ia_multiply"] = MagicMock()
    math_chain = CustomMathChain(mock_llm, mock_tools)

    class MockResponseMultipy:
        def __init__(self):
            self.content = None
            self.tool_calls = [{"name": "ia_multiply", "args": {"a": 5, "b": 2}}]

    mock_llm.bind_tools.return_value.invoke.return_value = MockResponseMultipy()
    mock_tools["ia_multiply"].invoke.return_value = 10

    response = math_chain.ask_question("How much is five times two?")
    assert response == [10]
    mock_llm.bind_tools.return_value.invoke.assert_called_once()
    mock_tools["ia_multiply"].invoke.assert_called_once_with({"a": 5, "b": 2})


def test_diff():
    mock_tools = MagicMock()
    mock_llm = MagicMock()
    mock_tools["ia_diff"] = MagicMock()
    math_chain = CustomMathChain(mock_llm, mock_tools)

    class MockResponseDiff:
        def __init__(self):
            self.content = None
            self.tool_calls = [{"name": "ia_diff", "args": {"a": 5, "b": 2}}]

    mock_llm.bind_tools.return_value.invoke.return_value = MockResponseDiff()
    mock_tools["ia_diff"].invoke.return_value = 3

    response = math_chain.ask_question("How much is five minus two?")
    assert response == [3]
    mock_llm.bind_tools.return_value.invoke.assert_called_once()
    mock_tools["ia_diff"].invoke.assert_called_once_with({"a": 5, "b": 2})


def test_add():
    mock_tools = MagicMock()
    mock_llm = MagicMock()
    mock_tools["ia_sum"] = MagicMock()
    math_chain = CustomMathChain(mock_llm, mock_tools)

    class MockResponse:
        def __init__(self):
            self.content = None
            self.tool_calls = [{"name": "ia_multiply", "args": {"a": 5, "b": 2}}]

    mock_llm.bind_tools.return_value.invoke.return_value = MockResponse()
    mock_tools["ia_add"].invoke.return_value = 7

    response = math_chain.ask_question("How much is five plus two?")
    assert response == [7]
    mock_llm.bind_tools.return_value.invoke.assert_called_once()
    mock_tools["ia_add"].invoke.assert_called_once_with({"a": 5, "b": 2})
