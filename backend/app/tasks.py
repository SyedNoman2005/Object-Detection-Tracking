import os
import asyncio
from celery import Celery
from app.core.config import settings
from app.services.video_processor import VideoProcessor

celery_app = Celery(
    "tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

@celery_app.task(bind=True)
def process_video_task(self, input_path: str, output_path: str):
    """Celery task to process video in background."""
    processor = VideoProcessor()
    
    def update_progress(current: int, total: int):
        self.update_state(state='PROGRESS', meta={'current': current, 'total': total})
        
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(processor.process_file(input_path, output_path, update_progress))
    return result
