import io
import json
import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import requests
from dotenv import load_dotenv

# Try optional ML imports
MODEL_AVAILABLE = False
model = None
transform = None
idx_to_label = None

try:
    import torch
    from torchvision import models, transforms
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI(title="NutriAyurveda Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FDC_API_KEY = os.getenv("FDC_API_KEY")
AYURVEDA_PATH = os.path.join(BASE_DIR, "ayurveda.json")
NUTRITION_FALLBACK = os.path.join(BASE_DIR, "nutrition_fallback.json")
IMAGENET_CLASSES = os.path.join(BASE_DIR, "imagenet_class_index.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Preload static data
AYURVEDA_DATA = load_json(AYURVEDA_PATH)
NUTRITION_DATA = load_json(NUTRITION_FALLBACK)

# Build a name->entry map with aliases
AYURVEDA_MAP = {}
for entry in AYURVEDA_DATA:
    AYURVEDA_MAP[entry["name"].lower()] = entry
    for a in entry.get("aliases", []):
        AYURVEDA_MAP[a.lower()] = entry


class AnalyzeResponse(BaseModel):
    item: str
    confidence: Optional[float] = None
    nutrition: dict
    ayurveda: Optional[dict] = None
    source: dict
    top_predictions: Optional[List[Dict[str, Any]]] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ml_model": MODEL_AVAILABLE,
        "usda": bool(FDC_API_KEY),
        "version": app.version,
    }


@app.get("/foods")
def foods():
    # Return a deduped sorted list of known foods to help the UI suggest names
    keys = set(NUTRITION_DATA.keys()) | set(AYURVEDA_MAP.keys())
    # Only base names (avoid duplicates due to aliases mapping to the same entry)
    return {"foods": sorted(keys)}


def init_model():
    global model, transform, idx_to_label
    if not MODEL_AVAILABLE:
        return False
    if model is not None:
        return True
    # Lightweight pretrained model (MobileNetV3 small)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()
    preprocess = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
    transform = preprocess
    # Fallback to imagenet labels bundled with torchvision
    try:
        import torchvision
        idx_to_label = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]
    except Exception:
        idx_to_label = None
    return True


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    # 1) Parse image
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    predicted_label = None
    confidence = None
    source = {"classifier": "none"}
    top_predictions: List[Dict[str, Any]] = []

    # 2) Use provided name if present; else run classifier if available
    if name and name.strip():
        predicted_label = name.strip().lower()
        source["override"] = "form_name"
    elif init_model():
        try:
            input_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                k = min(5, probs.shape[-1])
                confs, idxs = torch.topk(probs, k=k)
                for i in range(k):
                    lab = idx_to_label[int(idxs[i])] if idx_to_label else f"class_{int(idxs[i])}"
                    top_predictions.append({"label": lab, "confidence": float(confs[i].item())})
                # choose top-1 initially
                predicted_label = top_predictions[0]["label"]
                confidence = top_predictions[0]["confidence"]
                source["classifier"] = "mobilenet_v3_small_imagenet"
                source["top_k"] = top_predictions
        except Exception as e:
            # Classifier failure shouldn't block the pipeline
            source["classifier_error"] = str(e)

    # 3) If classifier missing or generic label, try EXIF and filename hints
    if not predicted_label or predicted_label.startswith("class_"):
        # Try EXIF string contents for hints
        try:
            exif = getattr(img, "getexif", lambda: None)()
            if exif:
                exif_text = " ".join(
                    str(v) for v in exif.values() if isinstance(v, (str, bytes))
                )
                if exif_text:
                    text = exif_text.lower()
                    for key in list(NUTRITION_DATA.keys()) + list(AYURVEDA_MAP.keys()):
                        if key in text:
                            predicted_label = key
                            confidence = None
                            source["heuristic"] = "exif_match"
                            break
        except Exception:
            pass

    if not predicted_label or predicted_label.startswith("class_"):
        name_hint = os.path.splitext(image.filename or "")[0].lower()
        for key in list(NUTRITION_DATA.keys()) + list(AYURVEDA_MAP.keys()):
            if key in name_hint:
                predicted_label = key
                confidence = None
                source["heuristic"] = "filename_match"
                break

    if not predicted_label:
        # Still unknown
        raise HTTPException(status_code=422, detail="Could not identify the food item. Try a clearer image or include the name in the filename.")

    # Normalize name (map common ImageNet labels to food where possible)
    normalized = predicted_label.lower()
    # Simple mapping from ImageNet terms to foods
    alias_map = {
        "granny_smith": "apple",
        "banana": "banana",
        "rice": "rice",
        "custard_apple": "apple",
        "pomegranate": "pomegranate",
        "ice_cream": "yogurt",
        "yogurt": "yogurt",
        # Common ImageNet edible classes
        "orange": "orange",
        "lemon": "lemon",
        "pineapple": "pineapple",
        "strawberry": "strawberry",
        "pizza": "pizza",
        "hotdog": "hot dog",
        "hamburger": "burger",
        "bagel": "bagel",
        "broccoli": "broccoli",
        "cauliflower": "cauliflower",
        "cucumber": "cucumber",
        "carrot": "carrot",
        "eggplant": "eggplant",
        "spinach": "spinach",
    }
    normalized = alias_map.get(normalized.replace(" ", "_"), normalized)

    # If we have multiple predictions, prefer the first that maps to a known item
    if top_predictions:
        for cand in top_predictions:
            raw = cand["label"].lower().replace(" ", "_")
            mapped = alias_map.get(raw, cand["label"].lower())
            if mapped in AYURVEDA_MAP or mapped in NUTRITION_DATA:
                predicted_label = mapped
                confidence = cand["confidence"]
                break

    # 4) Fetch nutrition
    nutrition = await fetch_nutrition(normalized)

    # 5) Fetch ayurveda
    ayur = AYURVEDA_MAP.get(normalized)

    return AnalyzeResponse(
        item=normalized,
        confidence=confidence,
        nutrition=nutrition or {},
        ayurveda=ayur,
        source=source,
        top_predictions=top_predictions or None,
    )


async def fetch_nutrition(query: str) -> Optional[dict]:
    # Prefer USDA API if configured
    if FDC_API_KEY:
        try:
            # Search for the best match
            resp = requests.get(
                "https://api.nal.usda.gov/fdc/v1/foods/search",
                params={
                    "api_key": FDC_API_KEY,
                    "query": query,
                    "pageSize": 1,
                    "dataType": ["Survey (FNDDS)", "SR Legacy", "Branded"],
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("foods"):
                food = data["foods"][0]
                nutrients = {
                    (n.get("nutrientName") or n.get("nutrientName")): n.get("value")
                    for n in food.get("foodNutrients", [])
                    if n.get("value") is not None
                }
                return {
                    "source": "usda",
                    "fdcId": food.get("fdcId"),
                    "description": food.get("description"),
                    "brand": food.get("brandName"),
                    "serving": food.get("servingSize") and f"{food.get('servingSize')} {food.get('servingSizeUnit','')}".strip(),
                    "nutrients": nutrients,
                }
        except Exception as e:
            # Fall back silently
            pass

    # Fallback JSON
    if query in NUTRITION_DATA:
        return NUTRITION_DATA[query]

    # Try simple singularization/pluralization
    if query.endswith("s") and query[:-1] in NUTRITION_DATA:
        return NUTRITION_DATA[query[:-1]]

    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
