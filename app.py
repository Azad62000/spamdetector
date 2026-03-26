import os
import joblib
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory=".")

class PredictRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5

class Explanation(BaseModel):
    top_keywords: List[str]

class PredictResponse(BaseModel):
    label: str
    spam_probability: Optional[float]
    explanation: Optional[Explanation] = None

MODEL_PATH = os.path.join("models", "best_model.joblib")
model = None

def get_explanation(text: str, top_n: int = 5) -> Dict[str, List[str]]:
    """Extract top contributing words using TF-IDF weights."""
    try:
        tfidf = model.named_steps['tfidf']
        # Transform text to get TF-IDF weights for this specific input
        feature_matrix = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()
        
        # Get indices and scores for non-zero entries
        feature_index = feature_matrix.nonzero()[1]
        tfidf_scores = zip(feature_index, [feature_matrix[0, x] for x in feature_index])
        
        # Sort by score descending and pick top_n
        sorted_tfidf = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        top_keywords = [feature_names[i] for i, score in sorted_tfidf[:top_n]]
        
        return {"top_keywords": top_keywords}
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return {"top_keywords": []}

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
    start_time = datetime.now()
    try:
        # Get probability
        if hasattr(model.named_steps["clf"], "predict_proba"):
            proba = float(model.predict_proba([req.text])[0][1])
        else:
            proba = None
        
        # Apply threshold tuning
        threshold = req.threshold if req.threshold is not None else 0.5
        label = "spam" if (proba is not None and proba >= threshold) else "ham"
        
        # Get explanation
        explanation_data = get_explanation(req.text)
        
        # Logging
        conf_str = f"{proba:.4f}" if proba is not None else "N/A"
        logger.info(f"Prediction={label} | Confidence={conf_str} | Threshold={threshold} | Length={len(req.text)}")
        
        return PredictResponse(
            label=label, 
            spam_probability=proba,
            explanation=Explanation(**explanation_data)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...), threshold: Optional[float] = 0.5):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        if hasattr(model.named_steps["clf"], "predict_proba"):
            proba = float(model.predict_proba([text])[0][1])
        else:
            proba = None
            
        label = "spam" if (proba is not None and proba >= threshold) else "ham"
        explanation_data = get_explanation(text)
        
        conf_str = f"{proba:.4f}" if proba is not None else "N/A"
        logger.info(f"File={file.filename} | Prediction={label} | Confidence={conf_str} | Length={len(text)}")
        
        return {
            "label": label, 
            "spam_probability": proba, 
            "filename": file.filename,
            "explanation": explanation_data
        }
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
def home(request: Request, tab: Optional[str] = "text", example: Optional[str] = None):
    text = ""
    if example == "spam":
        text = "Congratulations! You have won a free iPhone. Click here to claim your prize!"
    elif example == "ham":
        text = "Hi, can we meet tomorrow at 10 AM for the project discussion?"
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "result": {"text": text} if text else None, 
        "tab": tab
    })

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(request: Request, text: str = Form(...), threshold: float = Form(0.5)):
    try:
        # Get probability
        if hasattr(model.named_steps["clf"], "predict_proba"):
            proba = float(model.predict_proba([text])[0][1])
        else:
            proba = None
        
        # Apply threshold tuning
        label = "spam" if (proba is not None and proba >= threshold) else "ham"
        
        # Get explanation
        explanation_data = get_explanation(text)
        
        # Logging
        conf_str = f"{proba:.4f}" if proba is not None else "N/A"
        logger.info(f"Form-Text: Prediction={label} | Confidence={conf_str} | Threshold={threshold} | Length={len(text)}")
        
        result = {
            "label": label,
            "spam_probability": proba,
            "explanation": explanation_data,
            "text": text
        }
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "tab": "text"})
    except Exception as e:
        logger.error(f"Form prediction error: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "result": {"error": str(e)}})

@app.post("/predict_file_form", response_class=HTMLResponse)
async def predict_file_form(request: Request, file: UploadFile = File(...), threshold: float = Form(0.5)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        if hasattr(model.named_steps["clf"], "predict_proba"):
            proba = float(model.predict_proba([text])[0][1])
        else:
            proba = None
            
        label = "spam" if (proba is not None and proba >= threshold) else "ham"
        explanation_data = get_explanation(text)
        
        conf_str = f"{proba:.4f}" if proba is not None else "N/A"
        logger.info(f"Form-File: {file.filename} | Prediction={label} | Confidence={conf_str} | Length={len(text)}")
        
        result = {
            "label": label,
            "spam_probability": proba,
            "filename": file.filename,
            "explanation": explanation_data,
            "text": text
        }
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "tab": "file"})
    except Exception as e:
        logger.error(f"File form prediction error: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "result": {"error": str(e)}})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)
