from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import os
import uvicorn
import asyncio
from webapp.inference import VirgoInference

# Initialize FastAPI App
app = FastAPI(title="Virgo Chat AI")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disable browser caching for static assets (dev convenience)
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response

app.add_middleware(NoCacheMiddleware)

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(BASE_DIR, "virgo_data_tokens", "virgo_tokenizer.json")

# =============================================
# Model Registry — add new checkpoints here
# =============================================
MODEL_REGISTRY = {
    "virgo_instruct": {
        "name": "Virgo Instruct",
        "file": "virgo_instruct.pt",
        "description": "Instruction-following model fine-tuned for prompt execution",
        "kaggle": "https://www.kaggle.com/models/punitkashyap2007/virgo-instruct",
    },
    "virgo_chat": {
        "name": "Virgo Chat",
        "file": "virgo_chat_best.pt",
        "description": "Multi-turn dialogue conversational alignment checkpoint",
        "kaggle": "https://www.kaggle.com/models/punitkashyap2007/virgo-chat",
    },
    "virgo_align": {
        "name": "Virgo Align V1.0",
        "file": "virgo_IFT_ep2.pt",
        "description": "Preference and fine-grained alignment release",
        "kaggle": "https://www.kaggle.com/models/punitkashyap2007/virgo-align-v1-0",
    },
    "virgo_base": {
        "name": "Virgo Base V1.0",
        "file": "virgo_IFT_ep1.pt",
        "description": "Raw pre-trained base model checkpoint (120M)",
        "kaggle": "https://www.kaggle.com/models/punitkashyap2007/virgo-base-v1-0",
    },
}

# Loaded engine cache  {model_id: VirgoInference}
_loaded_engines: dict[str, VirgoInference] = {}
_active_model_id: str = "virgo_instruct"  # default


def _load_engine(model_id: str) -> VirgoInference:
    """Load a model into the cache (or return the cached one)."""
    if model_id in _loaded_engines:
        return _loaded_engines[model_id]

    info = MODEL_REGISTRY[model_id]
    path = os.path.join(BASE_DIR, "trained_models", info["file"])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading model '{info['name']}' from {path} ...")
    engine = VirgoInference(model_path=path, tokenizer_path=TOKENIZER_PATH)
    _loaded_engines[model_id] = engine
    return engine


def _get_active_engine() -> VirgoInference | None:
    return _loaded_engines.get(_active_model_id)


# Load the default model on startup
print("Initializing Virgo inference engine...")
try:
    _load_engine(_active_model_id)
except Exception as e:
    print(f"Failed to load default model: {e}")


# =============================================
# Request / Response schemas
# =============================================
class ChatRequest(BaseModel):
    prompt: str
    history: list[dict] = []
    max_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    repetition_penalty: float = 1.10

class ChatResponse(BaseModel):
    response: str
    generation_time: float
    tokens_used: int = 0

class SwitchModelRequest(BaseModel):
    model_id: str


# =============================================
# Endpoints
# =============================================
@app.get("/api/models")
def list_models():
    """Return available models and which one is active."""
    models = []
    for mid, info in MODEL_REGISTRY.items():
        path = os.path.join(BASE_DIR, "trained_models", info["file"])
        models.append({
            "id": mid,
            "name": info["name"],
            "description": info["description"],
            "available": os.path.exists(path),
            "loaded": mid in _loaded_engines,
            "active": mid == _active_model_id,
        })
    return {"models": models, "active": _active_model_id}


@app.post("/api/switch-model")
async def switch_model(request: SwitchModelRequest):
    """Switch to a different model checkpoint."""
    global _active_model_id

    model_id = request.model_id
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    info = MODEL_REGISTRY[model_id]
    path = os.path.join(BASE_DIR, "trained_models", info["file"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {info['file']}")

    # Load in a thread if not cached (blocks for ~10s on CPU)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _load_engine, model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _active_model_id = model_id
    return {"status": "ok", "active": model_id, "name": info["name"]}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    engine = _get_active_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Model engine is not initialized.")

    try:
        import time
        start_time = time.time()
        # Run inference in a threadpool since it's blocking PyTorch code
        loop = asyncio.get_event_loop()
        output, tokens_used = await loop.run_in_executor(
            None,
            lambda: engine.generate(
                prompt=request.prompt,
                history=request.history,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty
            )
        )
        end_time = time.time()
        return ChatResponse(response=output, generation_time=round(end_time - start_time, 2), tokens_used=tokens_used)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    return {"status": "ok", "active_model": _active_model_id, "model_loaded": _get_active_engine() is not None}

@app.post("/api/stop")
def stop_generation():
    engine = _get_active_engine()
    if engine is not None:
        engine.stop_requested = True
    return {"status": "stopped"}


# Mount static files (Frontend)
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

if __name__ == "__main__":
    print("Starting Virgo Web App Server...")
    uvicorn.run("webapp.main:app", host="0.0.0.0", port=8000, reload=True)
