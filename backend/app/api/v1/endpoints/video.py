import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult

from app.core.config import settings
from app.schemas.video import VideoProcessingResponse, TaskStatusResponse
from app.tasks import process_video_task

router = APIRouter()

@router.post("/upload-video", response_model=VideoProcessingResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(settings.UPLOAD_DIR, input_filename)
    output_path = os.path.join(settings.OUTPUT_DIR, f"annotated_{input_filename}")

    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Dispatch celery task
    task = process_video_task.delay(input_path, output_path)

    return VideoProcessingResponse(task_id=task.id, message="Video processing started")

@router.get("/results/{task_id}", response_model=TaskStatusResponse)
async def get_results(task_id: str):
    task = AsyncResult(task_id)
    
    if task.state == 'PENDING':
        return TaskStatusResponse(task_id=task_id, status='Pending')
    elif task.state != 'FAILURE':
        return TaskStatusResponse(
            task_id=task_id, 
            status=task.state, 
            result=task.info if isinstance(task.info, dict) else {"result": task.info}
        )
    else:
        return TaskStatusResponse(task_id=task_id, status='Failure', result={"error": str(task.info)})
