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
        "name": "Virgo Instruct Tuning",
        "file": "virgo_instruct_tuning.pt",
        "description": "Final instruction-tuning alignment checkpoint",
        "kaggle": "https://www.kaggle.com/models/punitkashyap2007/virgo-instruct",
    },
    "virgo_instruct_v2": {
        "name": "Virgo Instruct V2",
        "file": "virgo_instruction_v2.pt",
        "description": "Instruction-following model fine-tuned for prompt execution (V2)",
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
        alt_path = os.path.join(BASE_DIR, "trained_models", "final_models", info["file"])
        if os.path.exists(alt_path):
            path = alt_path
        else:
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
    top_p: float = 0.9
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


@app.get("/report", response_class=HTMLResponse)
def get_report_page():
    report_path = os.path.join(BASE_DIR, "docs", "PROJECT_REPORT.md")
    if not os.path.exists(report_path):
        report_path = os.path.join(BASE_DIR, "PROJECT_REPORT.md")
    
    content = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# Technical Report Not Found"

    # Escape backslashes for JS template string embedding
    js_content = content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Virgo SLM V1.0 - Technical Report</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.0/github-markdown-dark.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body {{
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        .header-bar {{
            background: rgba(13, 17, 23, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid #30363d;
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 14px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-bar a {{
            color: #58a6ff;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
        }}
        .header-bar a:hover {{ text-decoration: underline; }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
            padding: 40px 24px 80px 24px;
        }}
        .markdown-body {{
            background: transparent !important;
            box-sizing: border-box;
        }}
    </style>
</head>
<body>
    <div class="header-bar">
        <div><strong style="color: #f0f6fc; font-size: 16px;">♍ Virgo SLM V1.0 Technical Report</strong></div>
        <div>
            <a href="/">← Return to Web Studio</a> &nbsp;|&nbsp; 
            <a href="https://github.com/punitxdev/VIRGO-SLM" target="_blank">View on GitHub ↗</a>
        </div>
    </div>
    <div class="container">
        <article id="content" class="markdown-body"></article>
    </div>
    <script>
        const markdownText = `{js_content}`;
        document.getElementById('content').innerHTML = marked.parse(markdownText);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# Mount static files (Frontend)
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

if __name__ == "__main__":
    print("Starting Virgo Web App Server...")
    uvicorn.run("webapp.main:app", host="0.0.0.0", port=8000, reload=True)
