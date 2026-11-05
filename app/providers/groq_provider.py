import json
from typing import Any

from groq import AsyncGroq

from app.config import settings
from app.providers.base import (
    BaseLLMProvider,
    LLMInterpretationResult,
    ToolCall,
    manifest_to_tool_schema,
)

SYSTEM_PROMPT = """You are a multilingual voice intent interpreter for a blind or low-vision fintech Android app in Nigeria.
Your job is to parse the user's spoken utterance and invoke the matching tool from the available tools list.

CRITICAL RULES:
1. Utterances may be spoken in:
   - English / Nigerian English
   - Broken English / Nigerian Pidgin (e.g. "abeg send 5k give David", "wetin be my balance")
   - Yoruba (e.g. "Ẹ jọọ ẹ fi 5k ranṣẹ si David", "kí ni balance mi")
   - Igbo (e.g. "Biko ziga 5k nye David", "kedu balance m")
   - Hausa (e.g. "Don Allah tura 5k zuwa ga David", "nawa ne kudi na")
   - French (e.g. "S'il vous plaît envoyez 5k à David", "quel est mon solde")
   - German (e.g. "Bitte überweise 5k an David", "wie ist mein Kontostand")
   - Spanish (e.g. "Por favor envía 5k a David", "¿cuál es mi saldo?")
2. For transfer amounts:
   - Extract the EXACT raw text spoken by the user into 'amount_raw' (e.g. "5k", "five thousand naira", "ẹgbẹrún márùn-ún", "puku ise", "dubu biyar", "cinq mille", "fünftausend", "cinco mil", "20 grand", "1.5m", "N5000").
   - If you can determine the numeric value in Naira, provide it in 'amount_numeric_estimate' (e.g. 5000 for 5k). If unsure, leave it null.
   - NEVER invent or guess an amount. If the user did not say an amount, do NOT guess.
3. For recipient:
   - Extract ONLY the explicitly spoken recipient name into 'recipient' (e.g. "David", "Mama", "Chinedu", "Sarah").
   - NEVER guess or invent a recipient. If none was specified, leave it empty or null.
4. If the request does not clearly match any available tool or is completely outside the banking app scope, do not call any tool.
"""


class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None

    async def interpret(
        self,
        utterance: str,
        manifest: list[dict[str, Any]],
        screen_descriptor: str | None = None,
        dialog_state: dict[str, Any] | None = None,
    ) -> LLMInterpretationResult:
        if not self.client:
            raise ValueError("Groq API key not configured")

        # Convert manifest to OpenAI / Groq tool schema
        tools = []
        for action in manifest:
            tname, tdesc, tparams = manifest_to_tool_schema(action)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tname,
                        "description": tdesc,
                        "parameters": tparams,
                    },
                }
            )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        user_context_parts = [f"Utterance: {utterance}"]
        if screen_descriptor:
            user_context_parts.append(f"Current Screen: {screen_descriptor}")
        if dialog_state:
            user_context_parts.append(f"Dialog Context: {json.dumps(dialog_state)}")

        messages.append({"role": "user", "content": "\n".join(user_context_parts)})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            tool_choice="auto",
            temperature=0.0,
        )

        choice = response.choices[0]
        tool_calls_out = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError, KeyError):
                    args = {}
                tool_calls_out.append(ToolCall(name=tc.function.name, arguments=args))

        # Groq Llama 3.3 outputs high quality tool calls. Set baseline confidence
        confidence = 0.95 if tool_calls_out else 0.5

        return LLMInterpretationResult(
            tool_calls=tool_calls_out,
            confidence=confidence,
            text_content=choice.message.content,
            raw_response=response,
        )
