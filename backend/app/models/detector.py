from typing import Tuple, List, Optional
import numpy as np
import ultralytics
from ultralytics import YOLO
import torch

from app.core.config import settings
from app.core.logger import logger

class YOLODetector:
    _model: Optional[ultralytics.YOLO] = None

    @classmethod
    def load(cls) -> ultralytics.YOLO:
        if cls._model is None:
            logger.info(f"Loading YOLO model from {settings.MODEL_PATH} on {settings.DEVICE}")
            try:
                cls._model = YOLO(settings.MODEL_PATH)
                cls._model.to(settings.DEVICE)
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise
        return cls._model

    @classmethod
    async def detect(cls, frame: np.ndarray, confidence: float = 0.35, iou: float = 0.45) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        model = cls.load()
        # Perform inference
        results = model(frame, verbose=False, conf=confidence, iou=iou)[0]
        
        if len(results.boxes) == 0:
            return np.array([]), np.array([]), []

        # Extract bounding boxes (xyxy), confidence scores, and class IDs
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int).tolist()
        
        return boxes, scores, classes
