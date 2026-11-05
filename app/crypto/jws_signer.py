import base64
import hashlib
import json
import sys
import time
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from jwcrypto import jwk, jws

from app.config import settings


def canonical_json(data: Any) -> str:
    """Returns deterministic canonical JSON string with sorted keys, no whitespace, and UTF-8 characters."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_params(params: dict[str, Any]) -> str:
    """Returns SHA-256 hex digest of canonical JSON parameters."""
    canonical = canonical_json(params)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class JWSSigner:
    """Manages Ed25519 key pair, JWS signing, and JWKS public key distribution.

    Uses RFC 8037 (CFRG Elliptic Curve Diffie-Hellman (ECDH) and Signatures in JSON Object Signing and Encryption).
    """

    def __init__(self, private_key_b64: str = "", kid: str = "voice-intent-key-1"):
        self.kid = kid
        if private_key_b64:
            raw_bytes = base64.b64decode(private_key_b64)
            self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(raw_bytes)
        else:
            if (
                settings.APP_ENV == "production"
                and not settings.DEBUG
                and "pytest" not in sys.modules
            ):
                raise RuntimeError(
                    "ED25519_PRIVATE_KEY_B64 must be configured in production to ensure key persistence across restarts and instances."
                )
            # Generate a fresh key pair
            self._private_key = ed25519.Ed25519PrivateKey.generate()

        self._public_key = self._private_key.public_key()

        # Prepare JWK for jwcrypto
        priv_bytes = self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # Base64URL encode x and d
        x_b64 = base64.urlsafe_b64encode(pub_bytes).decode("ascii").rstrip("=")
        d_b64 = base64.urlsafe_b64encode(priv_bytes).decode("ascii").rstrip("=")

        self.jwk_key = jwk.JWK(
            kty="OKP",
            crv="Ed25519",
            kid=self.kid,
            use="sig",
            alg="EdDSA",
            x=x_b64,
            d=d_b64,
        )

        self.public_jwk_dict = {
            "kty": "OKP",
            "crv": "Ed25519",
            "kid": self.kid,
            "use": "sig",
            "alg": "EdDSA",
            "x": x_b64,
        }

    def get_jwks(self) -> dict[str, Any]:
        """Returns standard JWKS dictionary with public keys."""
        return {"keys": [self.public_jwk_dict]}

    def sign_token(
        self,
        session_id: str,
        action: str,
        params: dict[str, Any],
        nonce: str,
        ttl_seconds: int = 60,
        now_ts: int | None = None,
        amr: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Mints a short-lived signed JWS token bound to:

        - action
        - params_hash (SHA-256 of canonical params)
        - session_id
        - single-use nonce
        - amr (authentication method reference for device audit trail, e.g. ["biometric"])
        - exp & iat
        """
        current_time = int(time.time()) if now_ts is None else now_ts
        expires_at = current_time + ttl_seconds

        payload_dict = {
            "iss": "accessai",
            "sub": session_id,
            "aid": action,
            "action": action,
            "ph": hash_params(params),
            "params_hash": hash_params(params),
            "params": params,
            "jti": nonce,
            "nonce": nonce,
            "iat": current_time,
            "exp": expires_at,
        }
        if amr:
            payload_dict["amr"] = amr

        token = jws.JWS(json.dumps(payload_dict))
        token.add_signature(
            self.jwk_key,
            alg="EdDSA",
            protected={"alg": "EdDSA", "typ": "JWT", "kid": self.kid},
        )
        compact_token = token.serialize(compact=True)
        return compact_token, payload_dict

    def verify_token(self, token_str: str) -> dict[str, Any]:
        """Verifies a compact JWS token against the public key.

        Returns the decoded payload if valid; raises ValueError if invalid or expired.
        """
        verifier = jws.JWS()
        verifier.deserialize(token_str, key=self.jwk_key)
        payload = json.loads(verifier.payload.decode("utf-8"))

        # Check expiration
        now = int(time.time())
        if payload.get("exp") and now > payload["exp"]:
            raise ValueError("Token has expired")

        return payload


# Global signer instance
signer = JWSSigner(
    private_key_b64=settings.ED25519_PRIVATE_KEY_B64,
    kid=settings.JWKS_KID,
)
