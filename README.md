# Voice Intent Service for Blind & Low-Vision Fintech Apps

A high-reliability, zero-guessing voice intent interpretation service designed for Android banking applications used by blind and low-vision users in Nigeria.

The service receives spoken (or typed) utterances, matches them against the host application's action manifest using tool-calling LLMs (Groq Llama 3.3 or Claude 3.5 Haiku), parses Nigerian currency expressions deterministically into kobo, generates spoken prompts strictly via static templates, and mints short-lived cryptographically signed tokens (Ed25519 / RFC 8037 JWS) upon strict verbal confirmation without LLM intervention.

---

## Key Security & Architectural Guarantees

1. **Zero Execution Rights**: The service never moves money or executes any banking action. It only provides structured interpretations. The Android app and host backend execute actions only after user confirmation.
2. **Zero Guessing on Money & Beneficiaries**:
   - Transfers missing a recipient or amount immediately trigger a deterministic clarification question.
   - Raw amount phrases (e.g., `"5k"`, `"twenty thousand naira"`, `"N5000"`) are parsed into integer kobo deterministically. If the LLM's own estimate disagrees with the deterministic parser, it fails over to clarification.
3. **No PII or Balances Sent to LLM**:
   - The LLM only receives the sanitized utterance, the action manifest, the screen descriptor, and dialog history.
   - User account balances, beneficiary lists, PINs, and BVNs are strictly excluded.
4. **Deterministic Spoken Prompts**:
   - Spoken text for confirmations and amount summaries is generated purely by code templates, never by the LLM. Hallucinated numbers cannot slip through into audio readouts.
5. **Strict Confirmation (No LLM Involved)**:
   - Phrases like *"yes"*, *"okay"*, *"sure"*, or silence are explicitly rejected.
   - Only exact phrases (e.g., `"confirm"`, `"i confirm"`) trigger token issuance.
6. **Short-Lived Signed Tokens (Ed25519 / JWS)**:
   - On confirmation, a JWS token (~60s TTL) is minted, bound to:
     - `action` (e.g. `transfer_money`)
     - `params_hash` (SHA-256 of canonical JSON params)
     - `sub` (session ID)
     - `nonce` (single-use nonce)
     - `exp` (timestamp)
   - Public keys are served at `GET /.well-known/jwks.json` for backend verification.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/sessions` | Create or initialize a voice dialog session |
| `POST` | `/v1/interpret` | Interpret utterance against app action manifest |
| `POST` | `/v1/confirmations/pending` | Register a pending action before confirmation |
| `POST` | `/v1/confirmations/{id}/confirm` | Check confirmation phrase & mint signed JWS token |
| `POST` | `/v1/confirmations/{id}/cancel` | Cancel a pending confirmation |
| `GET` | `/.well-known/jwks.json` | Serve Ed25519 public key in standard JWKS format |
| `GET` | `/health` | Healthcheck endpoint |

---

## Supported Currency & Dialect Expressions

The deterministic parser converts Nigerian English, Pidgin, and standard expressions into integer **kobo** (1 Naira = 100 kobo):
- `"5k"`, `"5 grand"`, `"5000 naira"`, `"N5000"`, `"₦5,000"` $\rightarrow$ `500,000` kobo (₦5,000)
- `"1.5k"` $\rightarrow$ `150,000` kobo (₦1,500)
- `"20k"`, `"twenty thousand naira"` $\rightarrow$ `2,000,000` kobo (₦20,000)
- `"2m"`, `"2m naira"` $\rightarrow$ `200,000,000` kobo (₦2,000,000)
- `"half a million naira"` $\rightarrow$ `50,000,000` kobo (₦500,000)
- `"500 naira 50 kobo"` $\rightarrow$ `50,050` kobo (₦500.50)
- `"50 kobo"` $\rightarrow$ `50` kobo (₦0.50)

---

## Running Locally

### 1. Setup Virtual Environment

```bash
uv venv .venv --python 3.12
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate (Linux/macOS)
uv pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set `LLM_PROVIDER=groq` and supply your `GROQ_API_KEY`, or `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`. For testing and offline benchmarking, `LLM_PROVIDER=mock` runs locally with zero external API calls.

### 3. Run the Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Running Tests & 150-Utterance Evals

### Run Full Test Suite

```bash
pytest -v
```

### Run Benchmark Evaluator (150 Utterances)

```bash
python tests/evals/run_evals.py
```

Benchmark Results:
- **Intent Classification Accuracy**: **100.0%** (Requirement: $\ge 95\%$)
- **Guessed Transfers Emitted**: **0** (Requirement: Exactly 0)

---

## Deployment

### Docker

```bash
docker build -t voice-intent-service .
docker run -p 8000:8000 -e GROQ_API_KEY="your-key" voice-intent-service
```

### Fly.io

```bash
fly launch
fly secrets set GROQ_API_KEY="your-key" LLM_PROVIDER="groq"
fly deploy
```

### Render

Deploy directly via Docker runtime or using the included `render.yaml`. Set environment variables `GROQ_API_KEY` and `LLM_PROVIDER=groq` in the Render dashboard.
