from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.api.schemas import (
    CancelRequest,
    CancelResponse,
    ConfirmRequest,
    ConfirmResponse,
    InterpretRequest,
    InterpretResponse,
    PendingConfirmationRequest,
    PendingConfirmationResponse,
    SessionCreateRequest,
    SessionResponse,
)
from app.crypto.jws_signer import signer
from app.db.database import (
    create_session,
    get_session,
)
from app.providers import get_llm_provider
from app.services.confirmation import ConfirmationService
from app.services.interpreter import InterpreterService, validate_defence_in_depth

router = APIRouter()


def get_interpreter_service() -> InterpreterService:
    provider = get_llm_provider()
    return InterpreterService(provider=provider)


@router.post("/v1/sessions", response_model=SessionResponse, status_code=201)
async def create_new_session(req: SessionCreateRequest):
    return await create_session(
        session_id=req.session_id,
        metadata=req.metadata,
        device_key=req.device_key,
        app_id=req.app_id,
        locale=req.locale,
    )


@router.post("/v1/interpret", response_model=InterpretResponse)
async def interpret_utterance(
    req: InterpretRequest,
    service: Annotated[InterpreterService, Depends(get_interpreter_service)],
):
    user_text = req.text or req.utterance or ""
    raw_manifest = req.manifest or req.actions or []
    raw_dialog = req.dialog or req.dialog_state or {}

    # Defence in depth (v1.1): Reject 400 bad_request on 10+ digit runs or screen element values
    is_valid, err_reason = validate_defence_in_depth(
        utterance=user_text,
        screen=req.screen,
        screen_descriptor=req.screen_descriptor,
        dialog_state=raw_dialog,
    )
    if not is_valid:
        return JSONResponse(
            status_code=400,
            content={
                "error": {"code": "bad_request", "message": err_reason},
                "detail": {"code": "bad_request", "message": err_reason},
            },
        )

    # Ensure session exists
    session = await get_session(req.session_id)
    if not session:
        await create_session(session_id=req.session_id, locale=req.locale)

    try:
        return await service.interpret_utterance(
            utterance=user_text,
            text=user_text,
            manifest=raw_manifest,
            actions=raw_manifest,
            screen_descriptor=req.screen_descriptor,
            screen=req.screen,
            dialog_state=raw_dialog,
            dialog=raw_dialog,
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": {"code": "bad_request", "message": str(e)},
                "detail": {"code": "bad_request", "message": str(e)},
            },
        )


@router.post(
    "/v1/confirmations", response_model=PendingConfirmationResponse, status_code=201
)
@router.post(
    "/v1/confirmations/pending",
    response_model=PendingConfirmationResponse,
    status_code=201,
)
async def create_pending_action(req: PendingConfirmationRequest):
    action_name = req.action_id or req.action
    if not action_name:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "bad_request",
                    "message": "Missing action or action_id",
                },
                "detail": "Missing action or action_id",
            },
        )
    return await ConfirmationService.create_pending(
        session_id=req.session_id,
        action=action_name,
        params=req.params,
    )


@router.post("/v1/confirmations/{id}/confirm", response_model=ConfirmResponse)
async def confirm_action(id: str, req: ConfirmRequest):
    auth_dict = req.auth.model_dump() if req.auth else None
    success, result, status_code = await ConfirmationService.process_confirm(
        confirmation_id=id,
        utterance=req.utterance,
        auth=auth_dict,
        method=req.method or "voice",
        session_id=req.session_id,
    )
    if not success:
        return JSONResponse(status_code=status_code, content=result)
    return result


@router.post("/v1/confirm", response_model=ConfirmResponse)
@router.post("/v1/confirmations/confirm", response_model=ConfirmResponse)
async def confirm_action_direct(req: ConfirmRequest):
    conf_id = req.confirmation_id or req.pending_id
    if not conf_id:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "bad_request",
                    "message": "Missing confirmation_id or pending_id in request body",
                },
                "detail": "Missing confirmation_id or pending_id",
            },
        )
    auth_dict = req.auth.model_dump() if req.auth else None
    success, result, status_code = await ConfirmationService.process_confirm(
        confirmation_id=conf_id,
        utterance=req.utterance,
        auth=auth_dict,
        method=req.method or "voice",
        session_id=req.session_id,
    )
    if not success:
        return JSONResponse(status_code=status_code, content=result)
    return result


@router.post("/v1/confirmations/{id}/cancel", response_model=CancelResponse)
async def cancel_action(id: str, req: CancelRequest):
    success, result, status_code = await ConfirmationService.process_cancel(
        confirmation_id=id,
        reason=req.reason or "user_cancelled",
    )
    if not success:
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.get("/.well-known/jwks.json")
async def get_jwks():
    """Serves the public key in JWKS format for token verification by the host backend."""
    return signer.get_jwks()
