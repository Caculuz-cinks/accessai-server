import re
from typing import Any

from app.parsers.amount_parser import parse_amount_to_kobo
from app.providers.base import BaseLLMProvider, LLMInterpretationResult, ToolCall


class MockLLMProvider(BaseLLMProvider):
    """Deterministic multilingual rule-based provider for evaluations and test suites.

    Supports English, Broken English (Pidgin), Yoruba, Igbo, Hausa, French, German, and Spanish.
    """

    async def interpret(
        self,
        utterance: str,
        manifest: list[dict[str, Any]],
        screen_descriptor: str | None = None,
        dialog_state: dict[str, Any] | None = None,
    ) -> LLMInterpretationResult:
        manifest_ids = {item.get("id") or item.get("name") for item in manifest}
        text = utterance.strip().lower()

        # 1. Check for balance inquiries in all 8 languages
        balance_patterns = [
            # English & Pidgin
            r"\b(balance|wetin be my balance|how much dey|check my account|check account|show balance|view balance|account balance|how much i get|how much remain|how much is in|how much is left|what do i have in my account|check my money|wetin remain|how much money dey)\b",
            # Yoruba: "kí ni balance mi", "iye owo tó wà", "ṣayẹwo àkọọ́lẹ̀"
            r"\b(kí ni balance|ki ni balance|iye owo|ṣayẹwo àkọọ́lẹ̀|sayewo akoolo|owo to wa|owo tó wà)\b",
            # Igbo: "kedu balance m", "ego ole fọdụrụ", "ego dị na akaụntụ"
            r"\b(kedu balance|ego ole fọdụrụ|ego ole foduru|ego di na akawntu|ego dị na akaụntụ)\b",
            # Hausa: "nawa ne kudi na", "duba balance", "kudin asusu"
            r"\b(nawa ne kudi|duba balance|kudin asusu|kudin da ke)\b",
            # French: "quel est mon solde", "consulter mon solde", "combien sur mon compte"
            r"\b(quel est mon solde|consulter mon solde|vérifier mon solde|combien sur mon compte|mon solde)\b",
            # German: "wie ist mein kontostand", "kontostand", "guthaben"
            r"\b(wie ist mein kontostand|kontostand prüfen|mein kontostand|guthaben abfragen|mein guthaben)\b",
            # Spanish: "cuál es mi saldo", "consultar saldo", "cuánto tengo en mi cuenta"
            r"\b(cuál es mi saldo|cual es mi saldo|consultar saldo|ver mi saldo|cuánto tengo en mi cuenta)\b",
        ]
        for pat in balance_patterns:
            if re.search(pat, text, re.IGNORECASE) and "get_balance" in manifest_ids:
                return LLMInterpretationResult(
                    tool_calls=[ToolCall(name="get_balance", arguments={})],
                    confidence=0.96,
                )

        # 2. Check for transfer requests across all 8 languages
        pending_action = None
        if dialog_state:
            pending_action = dialog_state.get("pending_intent") or dialog_state.get(
                "pending_action_id"
            )

        transfer_keywords = [
            # English & Pidgin: send, transfer, wire, pay, give, dash
            r"\b(send|transfer|wire|pay|give|dash)\b",
            # Yoruba: firanṣẹ, ránṣẹ́, ranse, ranṣẹ
            r"\b(firanṣẹ|firanse|ránṣẹ́|ranse|ranṣẹ|san)\b",
            # Igbo: ziga, bufee
            r"\b(ziga|bufee)\b",
            # Hausa: tura
            r"\b(tura)\b",
            # French: envoyer, envoyez, virer, transférer, transfere
            r"\b(envoyer|envoyez|virer|transférer|transfere)\b",
            # German: überweise, uberweise, senden, sende
            r"\b(überweise|uberweise|senden|sende|überweisen)\b",
            # Spanish: enviar, envía, envia, transfiere, transferir
            r"\b(enviar|envía|envia|transfiere|transferir)\b",
        ]
        is_transfer = any(
            re.search(pat, text, re.IGNORECASE) for pat in transfer_keywords
        ) or (pending_action == "transfer_money")

        if is_transfer and "transfer_money" in manifest_ids:
            args: dict[str, Any] = {}

            # Pattern matching for amount extraction across languages
            amount_patterns = [
                # Compound naira and kobo e.g. 500 naira 50 kobo
                r"(\d+\s*naira\s*(?:and\s*)?\d+\s*kobo)",
                # Special fractions like half a million (English/French/German/Spanish)
                r"\b(half\s+(?:a\s+)?(?:million|billion|thousand)|moitié\s+d[' ]un\s+million|hälfte\s+einer\s+million|mitad\s+de\s+un\s+millón)\b(?:\s*naira)?",
                # Shorthand notation like 5k, 1.5k, 20 grand, 2m, etc.
                r"(?:₦|n|ngn|#)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|grand|m|million|b|billion)\b(?:\s*nairas?)?",
                # Currency symbols like N5000, ₦5,000, #5000
                r"(?:₦|n|ngn|#)\s*(\d+(?:,\d+)*(?:\.\d+)?)",
                # Digits with naira / kobo / nairas
                r"\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:nairas?|kobo)\b",
                # Multilingual word amounts (English, Yoruba, Igbo, Hausa, French, German, Spanish)
                r"\b(ẹgbẹrún\s+[a-z\-]+|egberun\s+[a-z\-]+|puku\s+[a-z]+|dubu\s+[a-z]+|cinq\s+mille|dix\s+mille|vingt\s+mille|fünftausend|zehntausend|cinco\s+mil|diez\s+mil|veinte\s+mil)\b(?:\s*nairas?)?",
                # English words e.g. twenty thousand naira
                r"\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|\s|-)+)\s*(?:nairas?|kobo)\b",
            ]

            raw_amount_str = None
            for pat in amount_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    candidate = m.group(0).strip()
                    kobo_test = parse_amount_to_kobo(candidate)
                    if kobo_test is not None and kobo_test > 0:
                        raw_amount_str = candidate
                        break

            # Check for bare numbers after transfer verbs (e.g. 'send 5000 to David')
            if not raw_amount_str:
                bare_m = re.search(
                    r"\b(?:send|transfer|wire|pay|give|dash|firanṣẹ|firanse|ránṣẹ́|ranse|ziga|tura|envoyer|virer|überweise|uberweise|enviar)\s+(\d{2,}(?:,\d+)*(?:\.\d+)?)(?!\s*(?:k|grand|m|million|b|billion|naira|kobo))\b",
                    text,
                    re.IGNORECASE,
                )
                if bare_m:
                    cand = bare_m.group(1).strip()
                    kobo_test = parse_amount_to_kobo(cand)
                    if kobo_test is not None and kobo_test > 0:
                        raw_amount_str = cand

            if raw_amount_str:
                args["amount_raw"] = raw_amount_str
                kobo_val = parse_amount_to_kobo(raw_amount_str)
                if kobo_val is not None:
                    args["amount_numeric_estimate"] = float(kobo_val / 100)

            # Recipient extraction across languages
            recip_patterns = [
                # English / Pidgin: to David, give David, dash David, for David
                r"\b(?:to|give|dash|for)\s+([a-zA-Z]+)(?:\s|$)",
                r"\b(?:send|transfer|wire|pay)\s+([a-zA-Z]+)\s+(?:₦|n|\d+|some|money)",
                # Yoruba: si David, fún David
                r"\b(?:si|fún|fun)\s+([a-zA-Z]+)(?:\s|$)",
                # Igbo: nye David, maaka David
                r"\b(?:nye|maaka)\s+([a-zA-Z]+)(?:\s|$)",
                # Hausa: zuwa ga David, zuwa David, wa David
                r"\b(?:zuwa\s+ga|zuwa|wa)\s+([a-zA-Z]+)(?:\s|$)",
                # French: à David, a David, pour David
                r"\b(?:à|a|pour)\s+([a-zA-Z]+)(?:\s|$)",
                # German: an David, für David
                r"\b(?:an|für|fur)\s+([a-zA-Z]+)(?:\s|$)",
                # Spanish: a David, para David
                r"\b(?:a|para)\s+([a-zA-Z]+)(?:\s|$)",
            ]
            recipient = None
            stopwords = {
                "money",
                "cash",
                "naira",
                "nairas",
                "kobo",
                "balance",
                "him",
                "her",
                "them",
                "me",
                "my",
                "account",
                "owo",
                "owó",
                "ego",
                "kudi",
                "kuɗi",
                "l'argent",
                "argent",
                "solde",
                "compte",
                "geld",
                "kontostand",
                "dinero",
                "cuenta",
                "saldo",
                # Numbers and scales should never be recipients
                "million",
                "billion",
                "thousand",
                "hundred",
                "half",
                "m",
                "k",
                "b",
                "grand",
                "dubu",
                "puku",
                "egberun",
                "ẹgbẹrún",
                "mille",
                "mil",
                "cent",
                "hundert",
                "cien",
            }
            for pat in recip_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    cand = m.group(1).strip()
                    if cand.lower() not in stopwords:
                        recipient = cand.capitalize()
                        break

            if not recipient and dialog_state:
                recipient = dialog_state.get("recipient") or dialog_state.get(
                    "filled_params", {}
                ).get("recipient")
            if recipient:
                args["recipient"] = recipient

            return LLMInterpretationResult(
                tool_calls=[ToolCall(name="transfer_money", arguments=args)],
                confidence=0.95,
            )

        # 3. Unknown tool or completely out of banking scope
        return LLMInterpretationResult(
            tool_calls=[],
            confidence=0.3,
            text_content="I could not identify a valid banking action for your request.",
        )
