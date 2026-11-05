from typing import Any

from pydantic import BaseModel, Field


class DeviceKey(BaseModel):
    kid: str
    alg: str = "ES256"
    public_key: str = Field(
        ...,
        description="Base64 X.509 SubjectPublicKeyInfo DER encoded ECDSA public key",
    )
    attestation_chain: list[str] | None = Field(
        default=None,
        description="Optional list of base64 DER X.509 certificates proving hardware backing",
    )


class SessionCreateRequest(BaseModel):
    session_id: str | None = None
    app_id: str | None = None
    locale: str | None = None
    device_key: DeviceKey | None = None
    metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    session_id: str
    status: str = "active"
    app_id: str | None = None
    locale: str | None = None
    device_key: DeviceKey | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: int
    updated_at: int


class ManifestParameterProperty(BaseModel):
    type: str
    description: str | None = None


class ManifestAction(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] | None = None


class InterpretRequest(BaseModel):
    session_id: str
    text: str | None = None
    utterance: str | None = None
    locale: str | None = "en-NG"
    actions: list[dict[str, Any]] | None = None
    manifest: list[dict[str, Any]] | None = None
    screen_descriptor: str | None = None
    screen: dict[str, Any] | list[Any] | str | None = None
    dialog: dict[str, Any] | None = None
    dialog_state: dict[str, Any] | None = None


class InterpretResponse(BaseModel):
    outcome: str = "action"  # "action" | "clarify" | "unsupported" | "control"
    type: str = "action"  # legacy alias for outcome
    action_id: str | None = None
    intent: str | None = None  # legacy alias for action_id
    params: dict[str, Any] = Field(default_factory=dict)
    missing_params: list[str] = Field(default_factory=list)
    control: str | None = None
    message: str = ""
    spoken_prompt: str = ""  # legacy alias for message
    confidence: float = 0.95
    clarification_reason: str | None = None
    language: str = "en"
    provider: str | None = None
    processing_region: str | None = None
    data_processing_notice: str | None = None


class PendingConfirmationRequest(BaseModel):
    session_id: str
    action_id: str | None = None
    action: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class PendingConfirmationResponse(BaseModel):
    pending_id: str
    confirmation_id: str
    session_id: str
    action_id: str
    action: str
    params: dict[str, Any]
    params_sha256: str
    params_hash: str
    status: str = "pending"
    spoken_prompt: str = ""
    message: str = ""
    expires_at: int
    expires_at_iso: str | None = None


class DeviceAuth(BaseModel):
    method: str = Field(
        default="biometric",
        description="Device unlock method: 'biometric' or 'device_credential'",
    )
    kid: str = Field(..., description="Key ID matching session device_key.kid")
    alg: str = Field(default="ES256", description="Signature algorithm (ES256)")
    signature: str = Field(
        ...,
        description="Base64 DER ECDSA signature over accessai-auth-v1|{session_id}|{pending_id}|{action_id}|{params_sha256}",
    )


class ConfirmRequest(BaseModel):
    session_id: str | None = None
    pending_id: str | None = None
    confirmation_id: str | None = None
    method: str = "voice"  # "voice" | "touch"
    utterance: str | None = None
    auth: DeviceAuth | None = None


class ConfirmResponse(BaseModel):
    status: str = "confirmed"
    confirmation_id: str
    pending_id: str
    token: str
    nonce: str
    expires_at: int
    spoken_prompt: str = ""
    message: str = ""


class CancelRequest(BaseModel):
    session_id: str | None = None
    reason: str | None = "user_cancelled"


class CancelResponse(BaseModel):
    status: str = "cancelled"
    confirmation_id: str
    spoken_prompt: str = ""
    message: str = ""
