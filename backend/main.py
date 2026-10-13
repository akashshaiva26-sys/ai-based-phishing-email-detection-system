"""
FastAPI Server for AI Phishing Email Detection System.
Provides REST APIs for text, image screenshot, and URL security analysis.
Serves the modern frontend dashboard as static assets.
"""

import os
import time
import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from detector.orchestrator import threat_orchestrator
from detector.ocr_vision import screenshot_analyzer
from detector.url_analyzer import url_analyzer
from detector.samples import SAMPLES

app = FastAPI(
    title="AI-Based Phishing Email Detection System",
    description="Multimodal Phishing Detection Engine with Image OCR, URL Intelligence, and NLP ML classification.",
    version="2.0.0"
)

# Enable CORS for local and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recent scan audit log (in-memory, last 50 scans)
SCAN_HISTORY = []

# Input Models
class TextAnalysisRequest(BaseModel):
    subject: Optional[str] = ""
    body: str
    sender: Optional[str] = ""
    reply_to: Optional[str] = ""
    raw_headers: Optional[str] = ""
    gemini_api_key: Optional[str] = ""

class ScreenshotBase64Request(BaseModel):
    image_base64: str
    client_ocr_text: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    mime_type: Optional[str] = "image/png"

class URLAnalysisRequest(BaseModel):
    url: str

def log_scan_history(analysis: dict, scan_type: str, subject_or_title: str):
    """Stores scan record in memory for history drawer."""
    record = {
        "id": f"scan_{int(time.time() * 1000)}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scan_type": scan_type,
        "title": subject_or_title[:60] if subject_or_title else "Email Inspection",
        "verdict": analysis.get("verdict", "UNKNOWN"),
        "risk_score": analysis.get("risk_score", 0.0),
        "badge_color": analysis.get("badge_color", "info")
    }
    SCAN_HISTORY.insert(0, record)
    if len(SCAN_HISTORY) > 50:
        SCAN_HISTORY.pop()

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "AI Phishing Email Detection Engine",
        "tesseract_backend": screenshot_analyzer.tesseract_available,
        "history_count": len(SCAN_HISTORY)
    }

@app.get("/api/samples")
def get_samples():
    return {"samples": SAMPLES}

@app.get("/api/history")
def get_history():
    return {"history": SCAN_HISTORY}

@app.post("/api/analyze/text")
def analyze_text_email(request: TextAnalysisRequest):
    if not request.body.strip() and not request.subject.strip():
        raise HTTPException(status_code=400, detail="Email body or subject must not be empty.")

    result = threat_orchestrator.analyze_email(
        body_text=request.body,
        subject=request.subject,
        sender=request.sender,
        reply_to=request.reply_to,
        raw_headers=request.raw_headers
    )
    title = request.subject or (request.body[:45] + "...")
    log_scan_history(result, "text_email", title)
    return result

@app.post("/api/analyze/image")
async def analyze_image_email(
    file: Optional[UploadFile] = File(None),
    client_ocr_text: Optional[str] = Form(""),
    gemini_api_key: Optional[str] = Form("")
):
    """Handles screenshot uploaded as multipart file."""
    if not file:
        raise HTTPException(status_code=400, detail="No image file uploaded.")

    content = await file.read()
    mime_type = file.content_type or "image/png"

    active_gemini_key = (gemini_api_key or "").strip() or os.getenv("GEMINI_API_KEY", "")

    screenshot_result = screenshot_analyzer.process_screenshot(
        image_bytes=content,
        client_ocr_text=client_ocr_text,
        gemini_api_key=active_gemini_key,
        mime_type=mime_type
    )

    if not screenshot_result.get("success"):
        raise HTTPException(status_code=400, detail=screenshot_result.get("error", "Failed to parse image."))

    extracted_text = screenshot_result.get("extracted_text", "")
    visuals = screenshot_result.get("visuals", {})
    gemini_data = screenshot_result.get("gemini_vision")

    # If OCR produced text, pass into full orchestrator
    full_analysis = threat_orchestrator.analyze_email(
        body_text=extracted_text,
        subject="[Screenshot Scan]",
        sender="",
        visual_metadata=visuals,
        gemini_result=gemini_data
    )

    # Attach screenshot metadata
    full_analysis["screenshot_meta"] = {
        "extracted_text": extracted_text,
        "ocr_source": screenshot_result.get("ocr_source"),
        "visuals": visuals
    }

    log_scan_history(full_analysis, "screenshot", file.filename or "Screenshot Upload")
    return full_analysis

@app.post("/api/analyze/image-base64")
def analyze_image_base64(request: ScreenshotBase64Request):
    """Handles screenshot pasted directly or captured via screen capture API."""
    try:
        raw_b64 = request.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        image_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    active_gemini_key = (request.gemini_api_key or "").strip() or os.getenv("GEMINI_API_KEY", "")

    screenshot_result = screenshot_analyzer.process_screenshot(
        image_bytes=image_bytes,
        client_ocr_text=request.client_ocr_text,
        gemini_api_key=active_gemini_key,
        mime_type=request.mime_type or "image/png"
    )

    if not screenshot_result.get("success"):
        raise HTTPException(status_code=400, detail=screenshot_result.get("error", "Failed to parse image."))

    extracted_text = screenshot_result.get("extracted_text", "")
    visuals = screenshot_result.get("visuals", {})
    gemini_data = screenshot_result.get("gemini_vision")

    full_analysis = threat_orchestrator.analyze_email(
        body_text=extracted_text,
        subject="[Live Screen Capture]",
        sender="",
        visual_metadata=visuals,
        gemini_result=gemini_data
    )

    full_analysis["screenshot_meta"] = {
        "extracted_text": extracted_text,
        "ocr_source": screenshot_result.get("ocr_source"),
        "visuals": visuals
    }

    log_scan_history(full_analysis, "screenshot_capture", "Live Screen Capture")
    return full_analysis

@app.post("/api/analyze/url")
def analyze_single_url(request: URLAnalysisRequest):
    """Deep analysis of a specific link/URL."""
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty.")
    return url_analyzer.analyze_single_url(request.url)

# Mount Frontend Static Assets
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    for sub in ["css", "js", "assets"]:
        sub_dir = os.path.join(FRONTEND_DIR, sub)
        if os.path.exists(sub_dir):
            app.mount(f"/{sub}", StaticFiles(directory=sub_dir), name=sub)

@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AI Phishing Email Detection API is running. Place index.html in frontend folder."}
