from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
from pathlib import Path
from context import get_and_rag
from upload import ingest_document



app = FastAPI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

# Pydantic model for prompt input
class PromptInput(BaseModel):
    prompt: str
    language: str
    voice: bool | None = False
    no_of_scenes: str | None = None

# # Endpoint to receive prompt and parameters
# @app.post("/submit-prompt/")
# async def submit_prompt(data: PromptInput):
#     try:
#         prompt = data.prompt

#         context_prompt = get_and_rag(prompt)

#         success = animator(context_prompt,data.language,data.voice,data.no_of_scenes)
        
#         # Here you can process the prompt and parameters
#         # For now, we'll just return them
#         return {
#             "message": f"Prompt received successfully:{success}",
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing prompt: {str(e)}")

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
    filename = "output.mp4"
    video_path = VIDEO_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=video_path,
        filename="output.mp4",
        media_type='video/mp4'
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


