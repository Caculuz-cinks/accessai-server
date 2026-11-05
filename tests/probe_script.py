"""Probe the AccessAI service: (A) exactly as the Android SDK v0.1.0 calls it, (B) its own API's safety behaviour."""

import base64
import hashlib
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.db.database import init_db
from app.main import app

SDK_HASH = "b729a280eabf6c63b6a6b889bfa1b26d4dad23286f2e3a36b98028e35879cd47"
SDK_MANIFEST = [
    {
        "id": "get_balance",
        "name": "check your balance",
        "description": "Read the user's available balance",
        "sensitivity": "read",
        "requires_confirmation": False,
        "params": [],
        "examples": ["what is my balance"],
    },
    {
        "id": "transfer_money",
        "name": "send money",
        "description": "Send money to a saved beneficiary",
        "sensitivity": "sensitive",
        "requires_confirmation": True,
        "params": [
            {
                "name": "recipient",
                "type": "string",
                "description": "Beneficiary name as the user said it",
                "required": True,
            },
            {
                "name": "amount_minor",
                "type": "amount_minor",
                "description": "Amount in kobo",
                "required": True,
            },
        ],
        "examples": ["send 5000 naira to David"],
    },
]

NATIVE_MANIFEST = [
    {"name": "get_balance", "description": "Get balance"},
    {
        "name": "transfer_money",
        "description": "Send money",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "amount_raw": {"type": "string"},
                "amount_numeric_estimate": {"type": "number"},
            },
        },
    },
]


def b64json(seg):
    return json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))


@pytest.mark.asyncio
async def test_probe_suite_23_checks():
    settings.LLM_PROVIDER = "mock"
    await init_db()
    results = []

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        async def call(method, path, body=None):
            headers = {
                "Content-Type": "application/json",
                "X-AccessAI-App": "accesspay-demo",
            }
            if method == "POST":
                res = await client.post(path, json=body, headers=headers)
            elif method == "GET":
                res = await client.get(path, headers=headers)
            else:
                raise ValueError(method)
            try:
                data = res.json()
            except (ValueError, TypeError, json.JSONDecodeError):
                data = res.text
            return res.status_code, data

        def check(name, ok, detail=""):
            results.append((name, ok))
            clean_name = name.encode("ascii", "replace").decode("ascii")
            clean_detail = str(detail).encode("ascii", "replace").decode("ascii")
            print(
                f"{'PASS' if ok else 'FAIL'}  {clean_name}"
                + (f"\n      {clean_detail}" if clean_detail else "")
            )

        print("\n=== A. As the Android SDK v0.1.0 calls it (contract v1) ===")
        # 1. POST /v1/sessions
        s, r = await call(
            "POST", "/v1/sessions", {"app_id": "accesspay-demo", "locale": "en-NG"}
        )
        check(
            "POST /v1/sessions returns session_id",
            s in (200, 201) and isinstance(r, dict) and "session_id" in r,
            f"{s} {str(r)[:160]}",
        )
        sid = r.get("session_id", "s-probe") if isinstance(r, dict) else "s-probe"

        # 2. POST /v1/interpret accepts SDK request
        body = {
            "session_id": sid,
            "text": "Send 5000 to David Adeyemi",
            "locale": "en-NG",
            "screen": {"screen_id": "home", "title": "Home", "elements": []},
            "actions": SDK_MANIFEST,
            "dialog": {
                "filled_params": {},
                "missing_params": [],
                "awaiting": "none",
            },
        }
        s, r = await call("POST", "/v1/interpret", body)
        check(
            "POST /v1/interpret accepts the SDK request",
            s == 200,
            f"{s} {str(r)[:300]}",
        )

        # 3. answers with outcome/action_id/params.amount_minor
        ok = (
            s == 200
            and isinstance(r, dict)
            and r.get("outcome") == "action"
            and r.get("action_id") == "transfer_money"
            and r.get("params", {}).get("amount_minor") == 500000
        )
        check(
            "…and answers with outcome/action_id/params.amount_minor", ok, str(r)[:300]
        )

        # 4. POST /v1/confirmations exists
        s, r = await call(
            "POST",
            "/v1/confirmations",
            {
                "session_id": sid,
                "action_id": "transfer_money",
                "params": {"amount_minor": 500000, "recipient": "ben_001"},
            },
        )
        check("POST /v1/confirmations exists", s in (200, 201), f"{s} {str(r)[:160]}")

        print("\n=== B. The service's own API (as implemented) ===")
        # 5. interpret Pidgin transfer
        s, r = await call(
            "POST",
            "/v1/interpret",
            {
                "session_id": "native-1",
                "utterance": "abeg send 5k give David",
                "manifest": NATIVE_MANIFEST,
            },
        )
        check(
            "interpret Pidgin transfer -> transfer_money, N5,000",
            s == 200
            and r.get("intent") == "transfer_money"
            and r.get("params", {}).get("amount_kobo") == 500000,
            str(r)[:220],
        )

        # 6. missing amount → clarification
        s, r = await call(
            "POST",
            "/v1/interpret",
            {
                "session_id": "native-1",
                "utterance": "send money to David",
                "manifest": NATIVE_MANIFEST,
            },
        )
        check(
            "missing amount → clarification (no guess)",
            s == 200 and r.get("type") == "clarification",
            str(r)[:200],
        )

        # 7. out of scope → no action
        s, r = await call(
            "POST",
            "/v1/interpret",
            {
                "session_id": "native-1",
                "utterance": "delete my account",
                "manifest": NATIVE_MANIFEST,
            },
        )
        check(
            "out of scope → no action",
            s == 200 and r.get("type") != "action",
            str(r)[:200],
        )

        # 8. Follow-up turn
        s, r = await call(
            "POST",
            "/v1/interpret",
            {
                "session_id": "native-1",
                "utterance": "5000 naira",
                "manifest": NATIVE_MANIFEST,
                "dialog_state": {
                    "pending_intent": "transfer_money",
                    "recipient": "David",
                },
            },
        )
        check(
            "follow-up '5000 naira' with dialog state continues the transfer",
            s == 200
            and r.get("intent") == "transfer_money"
            and r.get("type") == "action",
            str(r)[:220],
        )

        async def pending(sess, params):
            _s, _r = await call(
                "POST",
                "/v1/confirmations/pending",
                {"session_id": sess, "action": "transfer_money", "params": params},
            )
            return _r["confirmation_id"]

        # 9 - 13: strict confirmation phrases rejected
        cid = await pending(
            "native-1", {"amount_minor": 500000, "recipient": "ben_001"}
        )
        for phrase in [
            "yes",
            "okay",
            "don't confirm",
            "confirm transfer to David",
            "",
        ]:
            s, r = await call(
                "POST", f"/v1/confirmations/{cid}/confirm", {"utterance": phrase}
            )
            check(f"'{phrase}' does not confirm", s != 200, f"{s}")

        # 14. confirm transfer mints token
        s, r = await call(
            "POST",
            f"/v1/confirmations/{cid}/confirm",
            {"utterance": "confirm transfer"},
        )
        check(
            "'confirm transfer' mints a token",
            s == 200 and "token" in r,
            f"{s} {str(r)[:160]}",
        )

        # 15. token claims match contract
        if s == 200:
            _header, payload = (b64json(x) for x in r["token"].split(".")[:2])
            check(
                "token claims match contract (aid, ph, jti, iss=accessai)",
                all(k in payload for k in ("aid", "ph", "jti"))
                and payload.get("iss") == "accessai",
            )
            ph = payload.get("ph") or payload.get("params_hash")
            # 16. params hash matches SDK reference vector
            check(
                "params hash equals the SDK's canonical hash (reference vector)",
                ph == SDK_HASH,
                f"server={ph}",
            )
            from jwcrypto import jwk, jws

            keys = (await call("GET", "/.well-known/jwks.json"))[1]
            v = jws.JWS()
            v.deserialize(r["token"])
            try:
                v.verify(jwk.JWK(**keys["keys"][0]))
                check("token verifies against JWKS", True)
            except Exception as e:  # noqa: BLE001
                check("token verifies against JWKS", False, repr(e))

        # 18. second confirm refused
        s, r = await call(
            "POST",
            f"/v1/confirmations/{cid}/confirm",
            {"utterance": "confirm transfer"},
        )
        check("second confirm is refused (single use)", s != 200, f"{s} {str(r)[:120]}")

        # 19. session binding
        cid2 = await pending(
            "victim-session", {"amount_minor": 100, "recipient": "ben_001"}
        )
        s, r = await call(
            "POST", f"/v1/confirmations/{cid2}/confirm", {"utterance": "confirm"}
        )
        check(
            "confirm is bound to the owning session",
            s != 200,
            "confirmed without any session_id" if s == 200 else "",
        )

        # 20. Touch confirmation
        cid3 = await pending("native-1", {"amount_minor": 100, "recipient": "ben_001"})
        s, r = await call(
            "POST", f"/v1/confirmations/{cid3}/confirm", {"method": "touch"}
        )
        check(
            "Confirm button (method: touch) is supported",
            s == 200,
            f"{s} {str(r)[:160]}",
        )

        # 21. concurrent confirms mint at most one token
        cid4 = await pending("native-1", {"amount_minor": 100, "recipient": "ben_001"})
        import asyncio

        tasks = [
            call(
                "POST",
                f"/v1/confirmations/{cid4}/confirm",
                {"utterance": "confirm"},
            )
            for _ in range(8)
        ]
        responses = await asyncio.gather(*tasks)
        codes = [res[0] for res in responses]
        check(
            "concurrent confirms mint at most one token",
            codes.count(200) <= 1,
            f"status codes: {sorted(codes)}",
        )

        # 22. non-ASCII params hash (ensure_ascii=False)
        p = {"recipient": "Adébáyọ̀ Ọlá", "amount_minor": 500000}
        contract = hashlib.sha256(
            json.dumps(
                p, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        cid5 = await pending("native-1", p)
        s, r = await call(
            "POST", f"/v1/confirmations/{cid5}/confirm", {"utterance": "confirm"}
        )
        ph = b64json(r["token"].split(".")[1]).get("params_hash") if s == 200 else None
        check(
            "non-ASCII params hash the same as the SDK",
            ph == contract,
            f"server={ph}\n      contract={contract}",
        )

        # 23. errors use {error: {code, message}}
        s, r = await call(
            "POST",
            "/v1/confirmations/does-not-exist/confirm",
            {"utterance": "confirm"},
        )
        check(
            "errors use {error:{code,message}}",
            isinstance(r, dict)
            and isinstance(r.get("error"), dict)
            and "code" in r["error"],
            f"{s} {r}",
        )

        passed = sum(ok for _, ok in results)
        total = len(results)
        print(f"\n{passed}/{total} checks passed")
        assert passed == total, (
            f"Expected {total}/{total} checks to pass, got {passed}/{total}"
        )
