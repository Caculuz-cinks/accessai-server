import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


def manifest_to_tool_schema(action: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Extracts valid (tool_name, description, parameters_dict) from SDK or native manifest action.
    Sanitizes tool name to match ^[a-zA-Z0-9_-]+$ for LLM tool calling APIs (Groq, Anthropic, OpenAI).
    """
    raw_id = action.get("id") or action.get("name") or "action"
    tool_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id.strip())
    description = action.get("description") or action.get("name", "")

    if "parameters" in action and isinstance(action["parameters"], dict):
        params_schema = action["parameters"]
    elif "params" in action and isinstance(action["params"], list):
        properties = {}
        required = []
        for p in action["params"]:
            pname = p.get("name")
            if not pname:
                continue
            ptype = p.get("type", "string")
            if ptype == "amount_minor":
                json_type = "integer"
            elif ptype in ("number", "integer", "boolean", "string"):
                json_type = ptype
            else:
                json_type = "string"
            properties[pname] = {
                "type": json_type,
                "description": p.get("description", ""),
            }
            if p.get("required"):
                required.append(pname)
        params_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    else:
        params_schema = {"type": "object", "properties": {}}

    return tool_name, description, params_schema


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class LLMInterpretationResult(BaseModel):
    tool_calls: list[ToolCall] = []
    confidence: float = 1.0
    text_content: str | None = None
    raw_response: Any | None = None


class BaseLLMProvider(ABC):
    @abstractmethod
    async def interpret(
        self,
        utterance: str,
        manifest: list[dict[str, Any]],
        screen_descriptor: str | None = None,
        dialog_state: dict[str, Any] | None = None,
    ) -> LLMInterpretationResult:
        """
        Takes utterance, action manifest, screen descriptor, and dialog state.
        Never receives PII, balances, or beneficiary lists.
        Exposes manifest actions as tools and returns tool calls.
        """
