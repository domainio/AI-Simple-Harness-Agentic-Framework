import os

from pydantic import BaseModel, Field

from core.integrations.openwebui_adapter import run_chat, split_messages, stream_chat


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="")
        MODEL: str = "gpt-4o-mini"
        SYSTEM_PROMPT: str = (
            "You are a helpful agent. When creating or writing a file, always use an "
            "absolute path under /app/out/ (e.g. /app/out/<name>) unless the user gives "
            "an explicit absolute path."
        )
        MAX_STEPS: int = 8
        ENABLE_TOOLS: bool = True
        SHOW_STEPS: bool = True

    def __init__(self):
        self.valves = self.Valves()

    def pipe(self, body: dict):
        v = self.valves
        if v.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = v.OPENAI_API_KEY
        system, _, _ = split_messages(body["messages"], v.SYSTEM_PROMPT)
        runner = stream_chat if v.SHOW_STEPS else run_chat
        return runner(
            body["messages"],
            model_name=v.MODEL,
            system=system,
            max_steps=v.MAX_STEPS,
            enable_tools=v.ENABLE_TOOLS,
        )
