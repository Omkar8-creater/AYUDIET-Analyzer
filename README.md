# NutriAyurveda Analyzer

Identify a food from an image and show nutrition + Ayurvedic insights.

## Quick start (Windows, PowerShell)

1) Create and activate a virtual environment

```powershell
cd d:\SCANPLATE
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2) Install backend deps (FastAPI, etc.)

```powershell
pip install -r backend\requirements.txt
```

Note about Torch on Windows: this template uses a lightweight ImageNet model if available. Installing PyTorch CPU on Windows is best done from the official wheels. If installation via requirements fails, skip Torch for now—the app will still work using filename heuristics. Later, install CPU wheels from https://pytorch.org/get-started/locally/ and then `pip install torchvision`.

3) (Optional) Set USDA FoodData Central API key for live nutrition

```powershell
$env:FDC_API_KEY = "YOUR_FDC_API_KEY"
```

Without a key, the app falls back to `backend/nutrition_fallback.json` for common foods.

4) Run the API

```powershell
python backend\app.py
```

The server listens on http://localhost:8000

5) Open the frontend

- Open `frontend/index.html` in your browser, or
- Serve it statically (optional):

```powershell
# Simple Python HTTP server (optional)
python -m http.server 5500 -d frontend
```

Then visit http://localhost:5500

## API

POST /analyze (multipart/form-data)
- field: `image` — an image file

Response JSON
- item: normalized predicted name
- confidence: float (0-1) if model used
- nutrition: object (from USDA or fallback JSON)
- ayurveda: object from ayurveda.json if available
- source: info about classifier and fallbacks

GET /health
- status: ok
- ml_model: boolean
- usda: boolean
- version: API version

## Project layout

- backend/
  - app.py — FastAPI service
  - ayurveda.json — Ayurvedic mapping
  - nutrition_fallback.json — sample nutrition data
  - requirements.txt — Python deps
- frontend/
  - index.html, styles.css, app.js — simple UI

## Notes
- Classification uses MobileNetV3 Small (ImageNet) if Torch + TorchVision are installed. Otherwise the app will try filename hints.
- You can expand `ayurveda.json` with more foods and properties.
- Replace the nutrition fallback with your preferred database or enrich the USDA parsing for specific nutrients.
