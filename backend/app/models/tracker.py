from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

from app.core.config import settings
from app.core.logger import logger

class TrackingEvent:
    def __init__(self):
        self.total_count = 0
        self.tracked_objects = {}  # Store previous y-coordinates to detect crossing

class ObjectTracker:
    def __init__(self, max_age: int = 30):
        self.tracker = DeepSort(max_age=max_age, n_init=3, nms_max_overlap=1.0, embedder="mobilenet")
        self.event = TrackingEvent()

    def update(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: List[int], class_names: Dict[int, str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Format detections for Deep SORT: [([x, y, w, h], confidence, class), ...]
        detections = []
        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], score, cls_id))
            
        # Update tracks
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Calculate line position
        h, w = frame.shape[:2]
        line_y = int(h * settings.COUNTING_LINE_Y_RATIO)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
        
        tracking_info = {
            "active_tracks": 0,
            "people_counted": self.event.total_count,
            "alerts": False,
            "objects": []
        }
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cls_id = track.get_det_class()
            
            if cls_id is None:
                continue
                
            class_name = class_names.get(cls_id, str(cls_id))
            
            tracking_info["active_tracks"] += 1
            tracking_info["objects"].append({
                "id": track_id,
                "class": class_name,
                "bbox": [x1, y1, x2, y2]
            })
            
            # Draw bbox
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {track_id}", (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
            # Object counting logic (line crossing downwards)
            centroid_y = int((y1 + y2) / 2)
            if track_id in self.event.tracked_objects:
                prev_y = self.event.tracked_objects[track_id]
                if prev_y < line_y and centroid_y >= line_y:
                    self.event.total_count += 1
                    tracking_info["people_counted"] = self.event.total_count
                    if class_name == "person":
                        tracking_info["alerts"] = True
            
            self.event.tracked_objects[track_id] = centroid_y
            
        return frame, tracking_info
