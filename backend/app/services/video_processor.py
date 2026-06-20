import cv2
import os
import time
from typing import Optional, Dict, Any, Callable
from app.models.detector import YOLODetector
from app.models.tracker import ObjectTracker
from app.core.logger import logger

class VideoProcessor:
    def __init__(self):
        self.detector = YOLODetector()
        self.tracker = ObjectTracker()
        
    async def process_file(
        self, 
        input_path: str, 
        output_path: str, 
        progress_cb: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Process a video file and save the annotated result."""
        logger.info(f"Starting video processing: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        model = self.detector.load()
        class_names = model.names

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Inference
                boxes, scores, classes = await self.detector.detect(frame)
                
                # Tracking
                annotated_frame, tracking_info = self.tracker.update(frame, boxes, scores, classes, class_names)
                
                writer.write(annotated_frame)
                frame_idx += 1
                
                if progress_cb and frame_idx % 10 == 0:
                    progress_cb(frame_idx, total_frames)
                    
        finally:
            cap.release()
            writer.release()
            
        end_time = time.time()
        processing_time = end_time - start_time
        avg_fps = frame_idx / processing_time if processing_time > 0 else 0
        
        logger.info(f"Finished processing {frame_idx} frames in {processing_time:.2f}s ({avg_fps:.2f} FPS)")
        
        return {
            "frames_processed": frame_idx,
            "processing_time": processing_time,
            "average_fps": avg_fps,
            "people_counted": self.tracker.event.total_count
        }
