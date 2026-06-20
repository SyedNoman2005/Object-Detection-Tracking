from pydantic import BaseModel

class VideoProcessingResponse(BaseModel):
    task_id: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None
