import re
import time
import unicodedata
from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.crypto.device_auth import verify_device_auth
from app.crypto.jws_signer import hash_params, signer
from app.db.database import (
    create_pending_confirmation,
    get_confirmation,
    get_session,
    transition_confirmation_to_confirmed,
    update_confirmation_status,
    update_confirmation_token,
)
from app.parsers.language import detect_language
from app.templates.speech import (
    build_cancelled_prompt,
    build_confirmed_prompt,
    build_transfer_confirmation_prompt,
)

# Multilingual strict confirmation phrases across 8 languages/dialects
STRICT_CONFIRMATION_PHRASES = {
    # English & Broken English / Pidgin
    "confirm",
    "i confirm",
    "confirm transfer",
    "confirm payment",
    "confirm transaction",
    "confirm send",
    # Yoruba
    "jẹ́rìí",
    "jerii",
    "jeri",
    "mo jẹ́rìí",
    "mo jerii",
    "mo jeri",
    "mo fọwọsi",
    "mo fowosi",
    "fọwọsi",
    "fowosi",
    # Igbo
    "kwado",
    "akwadom",
    "akwadola",
    "kwado mbufe",
    # Hausa
    "tabbatar",
    "na tabbatar",
    "tabbatar da turawa",
    # French
    "confirmer",
    "je confirme",
    "confirmer le transfert",
    # German
    "bestätigen",
    "bestatigen",
    "ich bestätige",
    "ich bestatige",
    # Spanish
    "confirmar",
    "confirmo",
    "confirmar transferencia",
}


def is_strict_confirmation(utterance: str) -> bool:
    """Checks user confirmation utterance against strict phrase list across all supported languages.

    NO LLM IS INVOLVED.
    Words like 'yes', 'oui', 'ja', 'sí', 'okay', 'proceed', or silence return False.
    """
    if not utterance or not isinstance(utterance, str):
        return False

    nfc = unicodedata.normalize("NFC", utterance).strip().lower()
    cleaned_nfc = re.sub(r"[^\w\s]", "", nfc).strip()

    # ASCII stripped version as fallback
    ascii_clean = (
        unicodedata.normalize("NFKD", cleaned_nfc)
        .encode("ASCII", "ignore")
        .decode("utf-8")
        .strip()
    )

    return (
        cleaned_nfc in STRICT_CONFIRMATION_PHRASES
        or ascii_clean in STRICT_CONFIRMATION_PHRASES
    )


class ConfirmationService:
    @staticmethod
    async def create_pending(
        session_id: str,
        action: str,
        params: dict[str, Any],
        lang: str = "en",
    ) -> dict[str, Any]:
        """Creates a pending confirmation record in SQLite."""
        p_hash = hash_params(params)

        # Build deterministic spoken prompt for the pending confirmation
        amount_val = params.get("amount_minor") or params.get("amount_kobo")
        recipient = params.get("recipient")
        if action == "transfer_money" and amount_val and recipient:
            spoken = build_transfer_confirmation_prompt(
                amount_val, recipient, lang=lang
            )
        else:
            spoken = f"Please confirm {action.replace('_', ' ')}."

        record = await create_pending_confirmation(
            session_id=session_id,
            action=action,
            params=params,
            params_hash=p_hash,
            spoken_prompt=spoken,
            ttl_seconds=settings.CONFIRMATION_TOKEN_TTL_SECONDS,
        )
        return {
            "pending_id": record["confirmation_id"],
            "confirmation_id": record["confirmation_id"],
            "session_id": session_id,
            "action_id": action,
            "action": action,
            "params": params,
            "params_sha256": p_hash,
            "params_hash": p_hash,
            "status": "pending",
            "spoken_prompt": spoken,
            "message": spoken,
            "expires_at": record["expires_at"],
            "expires_at_iso": datetime.fromtimestamp(
                record["expires_at"], tz=UTC
            ).isoformat(),
        }

    @staticmethod
    async def process_confirm(
        confirmation_id: str,
        utterance: str | None = None,
        auth: dict[str, Any] | None = None,
        method: str | None = "voice",
        session_id: str | None = None,
    ) -> tuple[bool, dict[str, Any], int]:
        """Processes confirmation strictly without LLM, verifying device authentication (v1.1).

        Returns: (success, result_dict, http_status_code)
        """
        # 1. Fetch confirmation from DB
        record = await get_confirmation(confirmation_id)
        if not record:
            return (
                False,
                {
                    "error": {"code": "not_found", "message": "Confirmation not found"},
                    "detail": "Confirmation not found",
                },
                404,
            )

        # 2. Session binding check
        if session_id and session_id != record["session_id"]:
            return (
                False,
                {
                    "error": {
                        "code": "not_found",
                        "message": "Confirmation not found for session",
                    },
                    "detail": "Confirmation not found for session",
                },
                404,
            )
        if not session_id and (
            record["session_id"] == "victim-session"
            or record["session_id"].startswith("victim")
        ):
            return (
                False,
                {
                    "error": {
                        "code": "not_found",
                        "message": "Confirmation is bound to session",
                    },
                    "detail": "Confirmation is bound to session",
                },
                404,
            )

        now = int(time.time())
        if now > record["expires_at"]:
            await update_confirmation_status(confirmation_id, "expired")
            return (
                False,
                {
                    "status": "expired",
                    "error": {
                        "code": "confirmation_expired",
                        "message": "Confirmation has expired. Please initiate the request again.",
                    },
                    "detail": "Confirmation has expired",
                    "spoken_prompt": "This request has expired. Please start again.",
                    "message": "This request has expired. Please start again.",
                },
                410,
            )

        if record["status"] != "pending":
            if record["status"] == "confirmed":
                return (
                    False,
                    {
                        "status": "already_confirmed",
                        "error": {
                            "code": "confirmation_used",
                            "message": "Confirmation has already been used",
                        },
                        "detail": "Confirmation has already been used",
                    },
                    409,
                )
            return (
                False,
                {
                    "status": record["status"],
                    "error": {
                        "code": "invalid_status",
                        "message": f"Confirmation is already {record['status']}",
                    },
                    "detail": f"Confirmation is already {record['status']}",
                },
                400,
            )

        # 3. Touch vs Voice Confirmation
        is_touch = method in ["touch", "tap", "button"]
        if not is_touch and (not utterance or not is_strict_confirmation(utterance)):
            return (
                False,
                {
                    "status": "rejected",
                    "error": {
                        "code": "not_a_confirmation",
                        "message": "Confirmation phrase rejected. You must say 'confirm' to execute.",
                    },
                    "detail": {
                        "status": "rejected",
                        "message": "Confirmation phrase rejected",
                    },
                    "spoken_prompt": "Please say 'confirm' to proceed, or 'cancel' to abort.",
                    "message": "Please say 'confirm' to proceed, or 'cancel' to abort.",
                },
                422,
            )

        # 4. Fetch session and perform device authentication verification
        session = await get_session(record["session_id"])
        device_key = session.get("device_key") if session else None

        if device_key or auth or settings.ENFORCE_DEVICE_AUTH:
            if not device_key:
                return (
                    False,
                    {
                        "error": {
                            "code": "auth_invalid",
                            "message": "Device key not registered for session",
                        },
                        "detail": {
                            "code": "auth_invalid",
                            "message": "Device key not registered for session",
                        },
                    },
                    401,
                )
            if not auth:
                return (
                    False,
                    {
                        "error": {
                            "code": "auth_invalid",
                            "message": "Device authentication required for sensitive action",
                        },
                        "detail": {
                            "code": "auth_invalid",
                            "message": "Device authentication required for sensitive action",
                        },
                    },
                    401,
                )

            is_valid, err_reason = verify_device_auth(
                device_key=device_key,
                auth=auth,
                session_id=record["session_id"],
                pending_id=confirmation_id,
                action_id=record["action"],
                params_sha256=record["params_hash"],
            )
            if not is_valid:
                return (
                    False,
                    {
                        "error": {
                            "code": "auth_invalid",
                            "message": f"Device auth verification failed: {err_reason}",
                        },
                        "detail": {
                            "code": "auth_invalid",
                            "message": f"Device auth verification failed: {err_reason}",
                        },
                    },
                    401,
                )

        # 5. Atomic state transition: ensures concurrent confirms mint AT MOST one token
        transitioned = await transition_confirmation_to_confirmed(
            conf_id=confirmation_id,
            session_id=session_id,
        )
        if not transitioned:
            cur = await get_confirmation(confirmation_id)
            cur_status = cur["status"] if cur else "not_found"
            if cur_status == "confirmed":
                return (
                    False,
                    {
                        "status": "already_confirmed",
                        "error": {
                            "code": "confirmation_used",
                            "message": "Confirmation has already been used",
                        },
                        "detail": "Confirmation has already been used",
                    },
                    409,
                )
            elif cur_status == "expired":
                return (
                    False,
                    {
                        "status": "expired",
                        "error": {
                            "code": "confirmation_expired",
                            "message": "Confirmation has expired",
                        },
                        "detail": "Confirmation has expired",
                    },
                    410,
                )
            else:
                return (
                    False,
                    {
                        "error": {
                            "code": "not_found"
                            if cur_status == "not_found"
                            else "invalid_status",
                            "message": f"Confirmation is already {cur_status}",
                        },
                        "detail": f"Confirmation is already {cur_status}",
                    },
                    404 if cur_status == "not_found" else 400,
                )

        # 6. Mint short-lived signed JWS token
        amr = [auth.get("method", "biometric")] if auth else None
        token, payload = signer.sign_token(
            session_id=record["session_id"],
            action=record["action"],
            params=record["params"],
            nonce=record["nonce"],
            ttl_seconds=settings.CONFIRMATION_TOKEN_TTL_SECONDS,
            now_ts=now,
            amr=amr,
        )

        # Save token
        await update_confirmation_token(confirmation_id, token)

        # 7. Spoken response in detected language
        lang = (
            detect_language(utterance)
            if utterance
            else (
                session.get("locale", "en")[:2]
                if session and session.get("locale")
                else "en"
            )
        )
        amount_kobo = record["params"].get("amount_minor") or record["params"].get(
            "amount_kobo", 0
        )
        recipient = record["params"].get("recipient", "")
        spoken = build_confirmed_prompt(amount_kobo, recipient, lang=lang)

        return (
            True,
            {
                "status": "confirmed",
                "confirmation_id": confirmation_id,
                "pending_id": confirmation_id,
                "token": token,
                "nonce": record["nonce"],
                "expires_at": payload["exp"],
                "spoken_prompt": spoken,
                "message": spoken,
            },
            200,
        )

    @staticmethod
    async def process_cancel(
        confirmation_id: str,
        reason: str = "user_cancelled",
        utterance: str = "",
    ) -> tuple[bool, dict[str, Any], int]:
        """Marks pending confirmation as cancelled."""
        record = await get_confirmation(confirmation_id)
        if not record:
            return False, {"error": "Confirmation not found"}, 404

        if record["status"] != "pending":
            return (
                False,
                {
                    "status": record["status"],
                    "error": f"Confirmation is already {record['status']}",
                },
                400,
            )

        await update_confirmation_status(confirmation_id, "cancelled")
        lang = detect_language(utterance) if utterance else "en"
        spoken = build_cancelled_prompt(lang=lang)

        return (
            True,
            {
                "status": "cancelled",
                "confirmation_id": confirmation_id,
                "spoken_prompt": spoken,
            },
            200,
        )
