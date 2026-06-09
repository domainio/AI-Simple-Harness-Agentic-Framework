from pydantic import BaseModel

from agent_sdk.tools import ToolRegistry, to_openai_tool, tool
from agent_sdk.types import RunConfig


class EchoArgs(BaseModel):
    text: str


@tool(args=EchoArgs)
def echo(text: str) -> str:
    """Echo text."""
    return text


def test_tool_decorator_validates_and_invokes():
    assert echo.name == "echo"
    assert echo.description == "Echo text."
    assert echo.invoke({"text": "hi"}, RunConfig()) == "hi"


def test_tool_failures_return_error_data():
    out = echo.invoke({}, RunConfig())
    assert out.startswith("error: ValidationError:")


def test_registry_get_and_openai_schema():
    registry = ToolRegistry([echo])
    assert registry.get("echo") is echo
    schema = registry.openai_schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["additionalProperties"] is False


def test_to_openai_tool_shape():
    schema = to_openai_tool(echo)
    assert schema["function"]["description"] == "Echo text."
