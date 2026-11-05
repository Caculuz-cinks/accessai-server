import json
import re
from typing import Any

from app.api.schemas import InterpretResponse
from app.config import settings
from app.parsers.amount_parser import parse_amount_to_kobo, verify_amount_agreement
from app.parsers.language import detect_language
from app.providers.base import BaseLLMProvider
from app.templates.speech import (
    build_balance_prompt,
    build_clarification_prompt,
    build_transfer_confirmation_prompt,
)

RE_10_PLUS_DIGITS = re.compile(r"\d{10,}")
RE_VALUE_ATTR = re.compile(r'"value"\s*:\s*"(?!\[value\])[^"]+"', re.IGNORECASE)


def _contains_10_plus_digits(obj: Any) -> bool:
    if isinstance(obj, str):
        return bool(RE_10_PLUS_DIGITS.search(obj))
    if isinstance(obj, dict):
        return any(_contains_10_plus_digits(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_10_plus_digits(item) for item in obj)
    return False


def _has_element_value(obj: Any) -> bool:
    if isinstance(obj, dict):
        if "value" in obj and obj["value"] not in (None, ""):
            return True
        return any(_has_element_value(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_element_value(item) for item in obj)
    return False


def validate_defence_in_depth(
    utterance: str,
    screen: Any = None,
    screen_descriptor: str | None = None,
    dialog_state: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validates v1.1 defence-in-depth rules:

    1. Rejects any 10+ digit runs in utterance, screen, screen descriptor, or dialog state.
    2. Rejects any screen element containing a value.
    """
    if _contains_10_plus_digits(utterance):
        return (
            False,
            "10+ digit run detected in utterance; potential PIN, password, OTP, or account number leak",
        )

    if screen_descriptor and _contains_10_plus_digits(screen_descriptor):
        return False, "10+ digit run detected in screen descriptor"

    if screen and _contains_10_plus_digits(screen):
        return False, "10+ digit run detected in screen payload"

    if dialog_state and _contains_10_plus_digits(dialog_state):
        return False, "10+ digit run detected in dialog state"

    if screen and _has_element_value(screen):
        return (
            False,
            "Screen element contains a value; screen values must be filtered on device",
        )

    if screen_descriptor:
        try:
            parsed = json.loads(screen_descriptor)
            if _has_element_value(parsed):
                return (
                    False,
                    "Screen element contains a value; screen values must be filtered on device",
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            if RE_VALUE_ATTR.search(screen_descriptor):
                return (
                    False,
                    "Screen element contains a value; screen values must be filtered on device",
                )

    return True, ""


class InterpreterService:
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    def _build_response(
        self,
        outcome: str = "action",
        type: str = "action",
        confidence: float = 0.95,
        spoken_prompt: str = "",
        intent: str | None = None,
        action_id: str | None = None,
        params: dict[str, Any] | None = None,
        missing_params: list[str] | None = None,
        control: str | None = None,
        clarification_reason: str | None = None,
        language: str = "en",
    ) -> InterpretResponse:
        provider_name = settings.LLM_PROVIDER
        region = settings.LLM_REGION
        notice = (
            f"Processed by {provider_name} ({region}) pursuant to "
            f"Nigeria Data Protection Act 2023 (NDPA) cross-border guidelines."
        )
        final_action = action_id or intent
        final_intent = intent or action_id
        final_params = params or {}
        final_missing = missing_params or []
        msg = spoken_prompt

        return InterpretResponse(
            outcome=outcome,
            type=type,
            action_id=final_action,
            intent=final_intent,
            params=final_params,
            missing_params=final_missing,
            control=control,
            message=msg,
            spoken_prompt=msg,
            confidence=confidence,
            clarification_reason=clarification_reason,
            language=language,
            provider=provider_name,
            processing_region=region,
            data_processing_notice=notice,
        )

    async def interpret_utterance(
        self,
        utterance: str | None = None,
        manifest: list[dict[str, Any]] | None = None,
        screen_descriptor: str | None = None,
        dialog_state: dict[str, Any] | None = None,
        *,
        text: str | None = None,
        actions: list[dict[str, Any]] | None = None,
        screen: Any = None,
        dialog: dict[str, Any] | None = None,
    ) -> InterpretResponse:
        """Interprets a user utterance against the allowed actions/manifest across 8 languages.

        Supports Contract v1 (SDK format) and native format seamlessly.
        """
        # Support both positional (utterance, manifest, ...) and named (text, actions, ...) params
        user_text = (
            (utterance if isinstance(utterance, str) else None)
            or (text if isinstance(text, str) else None)
            or ""
        )
        user_text = user_text.strip()

        raw_manifest = (
            (manifest if isinstance(manifest, list) else None)
            or (actions if isinstance(actions, list) else None)
            or []
        )
        raw_dialog = (
            dialog
            if isinstance(dialog, dict)
            else (dialog_state if isinstance(dialog_state, dict) else {})
        )
        effective_screen = screen if screen is not None else screen_descriptor

        # Validate defence-in-depth rules
        is_valid, reason = validate_defence_in_depth(
            utterance=user_text,
            screen=effective_screen,
            screen_descriptor=screen_descriptor,
            dialog_state=raw_dialog,
        )
        if not is_valid:
            raise ValueError(reason)

        # Detect language
        lang = detect_language(user_text)

        # Build actions map
        manifest_map = {}
        for item in raw_manifest:
            aid = item.get("id") or item.get("name")
            if aid:
                manifest_map[aid] = item
            if item.get("name"):
                manifest_map[item["name"]] = item

        # Check for follow-up turns using dialog state
        pending_action = raw_dialog.get("pending_intent") or raw_dialog.get(
            "pending_action_id"
        )
        filled_params = raw_dialog.get("filled_params") or {}
        if not filled_params:
            filled_params = {
                k: v
                for k, v in raw_dialog.items()
                if k not in ["pending_intent", "pending_action_id", "awaiting"]
            }

        if pending_action == "transfer_money" or "transfer_money" in manifest_map:
            # Check if utterance answers a missing amount
            kobo_from_followup = parse_amount_to_kobo(user_text)
            existing_recipient = filled_params.get("recipient")
            if kobo_from_followup and kobo_from_followup > 0 and existing_recipient:
                verified_params = {
                    "recipient": existing_recipient,
                    "amount_raw": user_text,
                    "amount_kobo": kobo_from_followup,
                    "amount_minor": kobo_from_followup,
                }
                spoken_text = build_transfer_confirmation_prompt(
                    kobo_from_followup, existing_recipient, lang=lang
                )
                return self._build_response(
                    outcome="action",
                    type="action",
                    intent="transfer_money",
                    action_id="transfer_money",
                    params=verified_params,
                    missing_params=[],
                    confidence=0.96,
                    spoken_prompt=spoken_text,
                    language=lang,
                )

        # Ensure sanitized dialog state
        sanitized_dialog_state = None
        if raw_dialog:
            sanitized_dialog_state = {
                k: v
                for k, v in raw_dialog.items()
                if k
                not in [
                    "balance",
                    "beneficiaries",
                    "account_number",
                    "pin",
                    "bvn",
                    "password",
                    "last_system_message",
                    "value",
                ]
            }

        # Resolve screen descriptor
        desc = screen_descriptor
        if not desc and effective_screen:
            if isinstance(effective_screen, str):
                desc = effective_screen
            else:
                desc = json.dumps(effective_screen)

        llm_result = await self.provider.interpret(
            utterance=user_text,
            manifest=raw_manifest,
            screen_descriptor=desc,
            dialog_state=sanitized_dialog_state,
        )

        # 1. Check confidence threshold
        if llm_result.confidence < settings.CONFIDENCE_THRESHOLD:
            prompt = build_clarification_prompt("low_confidence", lang=lang)
            return self._build_response(
                outcome="unsupported",
                type="clarification",
                confidence=llm_result.confidence,
                spoken_prompt=prompt,
                clarification_reason="low_confidence",
                language=lang,
            )

        # 2. Check tool call count
        if len(llm_result.tool_calls) == 0:
            prompt = build_clarification_prompt("unknown_tool", lang=lang)
            return self._build_response(
                outcome="unsupported",
                type="clarification",
                confidence=llm_result.confidence,
                spoken_prompt=prompt,
                clarification_reason="unknown_tool",
                language=lang,
            )

        if len(llm_result.tool_calls) > 1:
            prompt = build_clarification_prompt("multiple_tools", lang=lang)
            return self._build_response(
                outcome="clarify",
                type="clarification",
                confidence=llm_result.confidence,
                spoken_prompt=prompt,
                clarification_reason="multiple_tools",
                language=lang,
            )

        tool_call = llm_result.tool_calls[0]
        tool_name = tool_call.name
        tool_args = tool_call.arguments or {}

        # 3. Check if tool is allowed
        if tool_name not in manifest_map:
            prompt = build_clarification_prompt("unknown_tool", lang=lang)
            return self._build_response(
                outcome="unsupported",
                type="clarification",
                confidence=llm_result.confidence,
                spoken_prompt=prompt,
                clarification_reason="unknown_tool",
                language=lang,
            )

        # 4. Handle money transfer validation
        if tool_name == "transfer_money":
            recipient = tool_args.get("recipient") or filled_params.get("recipient")
            if recipient:
                recipient = str(recipient).strip()
                if recipient.lower() in [
                    "someone",
                    "person",
                    "recipient",
                    "anybody",
                    "user",
                    "none",
                    "quelqu'un",
                    "jemand",
                    "alguien",
                ]:
                    recipient = None

            amount_raw = (
                tool_args.get("amount_raw")
                or tool_args.get("amount_minor")
                or filled_params.get("amount_raw")
                or filled_params.get("amount_minor")
            )
            if amount_raw is not None:
                amount_raw = str(amount_raw).strip()

            amount_kobo = None
            if amount_raw:
                amount_kobo = parse_amount_to_kobo(amount_raw)
                if amount_kobo is None and tool_args.get("amount_minor"):
                    try:
                        amount_kobo = int(tool_args["amount_minor"])
                    except (ValueError, TypeError):
                        pass

            # Missing info checks
            if not recipient and not amount_kobo:
                prompt = build_clarification_prompt(
                    "missing_amount_and_recipient", lang=lang
                )
                return self._build_response(
                    outcome="action",
                    type="clarification",
                    intent=tool_name,
                    action_id=tool_name,
                    params={},
                    missing_params=["recipient", "amount_minor"],
                    confidence=llm_result.confidence,
                    spoken_prompt=prompt,
                    clarification_reason="missing_amount_and_recipient",
                    language=lang,
                )

            if not recipient:
                prompt = build_clarification_prompt("missing_recipient", lang=lang)
                partial_params = {
                    "amount_kobo": amount_kobo,
                    "amount_minor": amount_kobo,
                    "amount_raw": amount_raw,
                }
                return self._build_response(
                    outcome="action",
                    type="clarification",
                    intent=tool_name,
                    action_id=tool_name,
                    params=partial_params,
                    missing_params=["recipient"],
                    confidence=llm_result.confidence,
                    spoken_prompt=prompt,
                    clarification_reason="missing_recipient",
                    language=lang,
                )

            if not amount_kobo:
                prompt = build_clarification_prompt(
                    "missing_amount", recipient=recipient, lang=lang
                )
                partial_params = {"recipient": recipient}
                return self._build_response(
                    outcome="action",
                    type="clarification",
                    intent=tool_name,
                    action_id=tool_name,
                    params=partial_params,
                    missing_params=["amount_minor"],
                    confidence=llm_result.confidence,
                    spoken_prompt=prompt,
                    clarification_reason="missing_amount",
                    language=lang,
                )

            # Cross-check against LLM's numeric estimate if provided
            llm_estimate = tool_args.get("amount_numeric_estimate")
            if llm_estimate is not None:
                try:
                    llm_num = float(llm_estimate)
                    if not verify_amount_agreement(amount_kobo, llm_num):
                        prompt = build_clarification_prompt(
                            "amount_mismatch", lang=lang
                        )
                        return self._build_response(
                            outcome="clarify",
                            type="clarification",
                            intent=tool_name,
                            action_id=tool_name,
                            params={"recipient": recipient},
                            missing_params=["amount_minor"],
                            confidence=llm_result.confidence,
                            spoken_prompt=prompt,
                            clarification_reason="amount_mismatch",
                            language=lang,
                        )
                except (ValueError, TypeError):
                    prompt = build_clarification_prompt("amount_mismatch", lang=lang)
                    return self._build_response(
                        outcome="clarify",
                        type="clarification",
                        intent=tool_name,
                        action_id=tool_name,
                        params={"recipient": recipient},
                        missing_params=["amount_minor"],
                        confidence=llm_result.confidence,
                        spoken_prompt=prompt,
                        clarification_reason="amount_mismatch",
                        language=lang,
                    )

            verified_params = {
                "recipient": recipient,
                "amount_raw": str(amount_raw or amount_kobo // 100),
                "amount_kobo": amount_kobo,
                "amount_minor": amount_kobo,
            }
            spoken_text = build_transfer_confirmation_prompt(
                amount_kobo, recipient, lang=lang
            )
            return self._build_response(
                outcome="action",
                type="action",
                intent=tool_name,
                action_id=tool_name,
                params=verified_params,
                missing_params=[],
                confidence=llm_result.confidence,
                spoken_prompt=spoken_text,
                language=lang,
            )

        # 5. Handle get_balance
        elif tool_name == "get_balance":
            spoken_text = build_balance_prompt(lang=lang)
            return self._build_response(
                outcome="action",
                type="action",
                intent=tool_name,
                action_id=tool_name,
                params={},
                missing_params=[],
                confidence=llm_result.confidence,
                spoken_prompt=spoken_text,
                language=lang,
            )

        # 6. Other generic manifest actions
        else:
            parsed_args = dict(tool_args)
            action_def = manifest_map.get(tool_name, {})
            # Look for amount_minor parameters in action params
            param_defs = action_def.get("params") or []
            for p in param_defs:
                p_name = p.get("name")
                if p.get("type") == "amount_minor" and p_name in parsed_args:
                    kobo_val = parse_amount_to_kobo(str(parsed_args[p_name]))
                    if kobo_val is not None:
                        parsed_args[p_name] = kobo_val
                        parsed_args["amount_minor"] = kobo_val
                        parsed_args["amount_kobo"] = kobo_val

            return self._build_response(
                outcome="action",
                type="action",
                intent=tool_name,
                action_id=tool_name,
                params=parsed_args,
                missing_params=[],
                confidence=llm_result.confidence,
                spoken_prompt=f"Performing {tool_name.replace('_', ' ')}.",
                language=lang,
            )
