from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import shutil
import time
import os
from pathlib import Path
from context import get_and_rag  # Commented out for testing
from upload import ingest_document  # Commented out for testing
from Animator import main_workflow
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# Dummy functions for testing when imports are not available

# from Animator import main_workflow  # Commented out for testing



app = FastAPI()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
print(f"Base directory: {BASE_DIR}")
print(f"Static directory: {STATIC_DIR}")
print(f"Static directory exists: {os.path.exists(STATIC_DIR)}")
print(f"Static directory contents: {os.listdir(STATIC_DIR) if os.path.exists(STATIC_DIR) else 'Directory not found'}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

# Pydantic model for prompt input
class PromptInput(BaseModel):
    prompt: str
    language: str
    voice: Optional[bool] = False
    no_of_scenes: Optional[str] = None


# Endpoint to receive prompt and parameters
@app.post("/submit-prompt/")
async def submit_prompt(data: PromptInput):
    try:
        prompt = data.prompt
        # Comment out RAG for now since we're in dummy mode
        # context_prompt = get_and_rag(prompt)

        success = "success"
        # main_workflow(context_prompt,data.language,data.voice,data.no_of_scenes)  # Original animator call
        dummy_result = main_workflow(prompt, data.language, data.voice, data.no_of_scenes)
        
        # Here you can process the prompt and parameters
        # For now, we'll just return them
        return {
            "message": f"Prompt received successfully (DUMMY MODE): {success}",
            "dummy_video_path": dummy_result,
            "note": "This is a dummy response for testing frontend connection"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prompt: {str(e)}")

# Endpoint to upload PDF
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try :
            ingest_document(file_path)
        except:
            raise HTTPException(status_code=400, detail="Only files are allowed")

        
        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")
    finally:
        file.file.close()

# Endpoint to download video
@app.get("/download-video")
async def download_video():
    try:
        video_path = Path("output/final_video.mp4")
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        
        return FileResponse(
            path=str(video_path),
            filename="generated_video.mp4",
            media_type='video/mp4'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Route to serve the landing page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    landing_page_path = os.path.join(STATIC_DIR, "imagio_hypnotic_spiral.html")
    print(f"Looking for landing page at: {landing_page_path}")
    
    if os.path.exists(landing_page_path):
        try:
            with open(landing_page_path, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content)
        except Exception as e:
            print(f"Error reading landing page: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading landing page: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail=f"Landing page not found at {landing_page_path}")

# Route to serve the main app page
@app.get("/app", response_class=HTMLResponse)
async def read_app():
    app_page_path = os.path.join(STATIC_DIR, "imagio_updated.html")
    print(f"Looking for app page at: {app_page_path}")
    
    if os.path.exists(app_page_path):
        try:
            with open(app_page_path, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content)
        except Exception as e:
            print(f"Error reading app page: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading app page: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail=f"App page not found at {app_page_path}")

# Route to serve static files
@app.get("/static/{path:path}")
async def serve_static(path: str):
    file_path = os.path.join(STATIC_DIR, path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail=f"File not found: {path}")



