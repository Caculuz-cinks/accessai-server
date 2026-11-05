import pytest
from httpx import ASGITransport, AsyncClient

from app.db.database import init_db
from app.main import app

MANIFEST = [
    {
        "name": "get_balance",
        "description": "Check current account balance",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "transfer_money",
        "description": "Transfer money to another account or person",
        "parameters": {
            "type": "object",
            "properties": {
                "amount_raw": {"type": "string"},
                "amount_numeric_estimate": {"type": "number"},
                "recipient": {"type": "string"},
            },
            "required": ["amount_raw", "recipient"],
        },
    },
]


@pytest.mark.asyncio
async def test_full_api_workflow():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

        # 2. GET /.well-known/jwks.json
        res = await client.get("/.well-known/jwks.json")
        assert res.status_code == 200
        jwks = res.json()
        assert "keys" in jwks
        assert len(jwks["keys"]) > 0
        assert jwks["keys"][0]["kty"] == "OKP"
        assert jwks["keys"][0]["crv"] == "Ed25519"
        assert jwks["keys"][0]["alg"] == "EdDSA"

        # 3. POST /v1/sessions
        res = await client.post(
            "/v1/sessions", json={"metadata": {"user": "test_blind_user"}}
        )
        assert res.status_code == 201
        session_data = res.json()
        session_id = session_data["session_id"]
        assert session_id is not None

        # 4. POST /v1/interpret (Valid transfer request in Nigerian English)
        res = await client.post(
            "/v1/interpret",
            json={
                "session_id": session_id,
                "utterance": "abeg send 5k give David",
                "manifest": MANIFEST,
                "screen_descriptor": "Home Screen",
            },
        )
        assert res.status_code == 200
        interp = res.json()
        assert interp["type"] == "action"
        assert interp["intent"] == "transfer_money"
        assert interp["params"]["amount_kobo"] == 500_000
        assert interp["params"]["recipient"] == "David"

        # 5. POST /v1/confirmations/pending
        res = await client.post(
            "/v1/confirmations/pending",
            json={
                "session_id": session_id,
                "action": interp["intent"],
                "params": interp["params"],
            },
        )
        assert res.status_code == 201
        pending = res.json()
        conf_id = pending["confirmation_id"]
        assert pending["status"] == "pending"

        # 6. POST /v1/confirmations/{id}/confirm with informal "yes" -> MUST BE REJECTED (422 not_a_confirmation)
        res = await client.post(
            f"/v1/confirmations/{conf_id}/confirm", json={"utterance": "yes"}
        )
        assert res.status_code == 422
        assert res.json()["error"]["code"] == "not_a_confirmation"

        # 7. POST /v1/confirmations/{id}/confirm with strict "confirm" -> SUCCESS
        res = await client.post(
            f"/v1/confirmations/{conf_id}/confirm", json={"utterance": "confirm"}
        )
        assert res.status_code == 200
        confirmed = res.json()
        assert confirmed["status"] == "confirmed"
        assert "token" in confirmed
        assert len(confirmed["token"].split(".")) == 3
