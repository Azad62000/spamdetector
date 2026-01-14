import os
import joblib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    spam_probability: Optional[float]

MODEL_PATH = os.path.join("models", "best_model.joblib")
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("models/best_model.joblib not found. Train offline and include the artifact.")
    model = joblib.load(MODEL_PATH)

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = float(model.predict_proba([req.text])[0][1])
    else:
        proba = None
    label = str(model.predict([req.text])[0])
    return PredictResponse(label=label, spam_probability=proba)

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = float(model.predict_proba([text])[0][1])
    else:
        proba = None
    label = str(model.predict([text])[0])
    return {"label": label, "spam_probability": proba, "filename": file.filename}

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("index.html")

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(text: str = Form(...)):
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = float(model.predict_proba([text])[0][1])
    else:
        proba = None
    label = str(model.predict([text])[0])
    return FileResponse("index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)
