# Oral Cancer Risk Prediction API

FastAPI backend for AI-assisted oral cancer screening support. This service uses structured risk-factor scoring plus optional voice abnormality scoring with decision-level fusion:

```text
Final Risk Score = 0.70 * Structured Risk Score + 0.30 * Voice Abnormality Score
```

This API is screening support only and is not a medical diagnosis.

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "oral-cancer-risk-api"
}
```

## Endpoints

- `POST /api/predict/risk-factors` accepts structured risk factors as JSON.
- `POST /api/predict/voice` accepts multipart form data with a `.wav` file field named `file`.
- `POST /api/predict/multimodal` accepts either JSON with `riskFactors` and optional `voiceScore`, or multipart form data with `riskFactors` as a JSON string plus optional `file` or `voiceScore`.
- `GET /api/history` returns in-memory prediction history for the current server process.

## Frontend API Base URL

Use `http://localhost:8000` for web.

Use `http://10.0.2.2:8000` for Android emulator.

Use your LAN IP for physical mobile device testing.

## Notes

- Dataset and model files are loaded from `../data_sets/arranged_dataset`.
- Models are loaded lazily and are not retrained during API startup.
- Uploaded audio is written to a temporary file for feature extraction and deleted after prediction.
- Missing model/script files return HTTP 503.
