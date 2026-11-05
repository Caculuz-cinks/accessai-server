import pytest

from app.crypto.jws_signer import hash_params, signer
from app.db.database import init_db
from app.services.confirmation import ConfirmationService, is_strict_confirmation


@pytest.mark.asyncio
async def test_strict_confirmation_phrases():
    # Only strict confirmation phrases pass
    assert is_strict_confirmation("confirm") is True
    assert is_strict_confirmation("Confirm") is True
    assert is_strict_confirmation("  confirm!  ") is True
    assert is_strict_confirmation("I confirm") is True
    assert is_strict_confirmation("confirm transfer") is True
    assert is_strict_confirmation("confirm payment") is True
    assert is_strict_confirmation("confirm transaction") is True

    # Critical requirement: 'Yes', 'okay', or silence DOES NOT COUNT
    assert is_strict_confirmation("yes") is False
    assert is_strict_confirmation("Yes, please") is False
    assert is_strict_confirmation("yeah") is False
    assert is_strict_confirmation("okay") is False
    assert is_strict_confirmation("ok") is False
    assert is_strict_confirmation("sure") is False
    assert is_strict_confirmation("proceed") is False
    assert is_strict_confirmation("go ahead") is False
    assert is_strict_confirmation("") is False
    assert is_strict_confirmation("   ") is False


@pytest.mark.asyncio
async def test_jws_minting_and_verification():
    await init_db()
    session_id = "sess-test-123"
    action = "transfer_money"
    params = {"recipient": "David", "amount_kobo": 500_000, "amount_raw": "5k"}
    nonce = "nonce-abc-456"

    token_str, _payload = signer.sign_token(
        session_id=session_id,
        action=action,
        params=params,
        nonce=nonce,
        ttl_seconds=60,
    )

    assert isinstance(token_str, str)
    assert len(token_str.split(".")) == 3  # Compact JWS format

    # Verify token payload
    verified_payload = signer.verify_token(token_str)
    assert verified_payload["sub"] == session_id
    assert verified_payload["action"] == action
    assert verified_payload["nonce"] == nonce
    assert verified_payload["params_hash"] == hash_params(params)
    assert verified_payload["params"]["amount_kobo"] == 500_000


@pytest.mark.asyncio
async def test_confirmation_lifecycle():
    await init_db()
    session_id = "sess-lifecycle-test"
    action = "transfer_money"
    params = {"recipient": "Amaka", "amount_kobo": 2_000_000, "amount_raw": "20k"}

    # 1. Create pending confirmation
    pending = await ConfirmationService.create_pending(
        session_id=session_id,
        action=action,
        params=params,
    )
    conf_id = pending["confirmation_id"]
    assert pending["status"] == "pending"

    # 2. Reject informal words like 'yes' (422 not_a_confirmation)
    success, res, status_code = await ConfirmationService.process_confirm(
        conf_id, "yes"
    )
    assert success is False
    assert status_code == 422
    assert res["status"] == "rejected"

    # 3. Confirm with strict phrase
    success, res, status_code = await ConfirmationService.process_confirm(
        conf_id, "confirm"
    )
    assert success is True
    assert status_code == 200
    assert res["status"] == "confirmed"
    assert "token" in res

    # 4. Token cannot be confirmed again (409 confirmation_used)
    success2, _res2, status_code2 = await ConfirmationService.process_confirm(
        conf_id, "confirm"
    )
    assert success2 is False
    assert status_code2 == 409


@pytest.mark.asyncio
async def test_confirmation_cancel():
    await init_db()
    session_id = "sess-cancel-test"
    action = "transfer_money"
    params = {"recipient": "Chinedu", "amount_kobo": 100_000, "amount_raw": "1k"}

    pending = await ConfirmationService.create_pending(
        session_id=session_id,
        action=action,
        params=params,
    )
    conf_id = pending["confirmation_id"]

    success, res, status_code = await ConfirmationService.process_cancel(conf_id)
    assert success is True
    assert status_code == 200
    assert res["status"] == "cancelled"
