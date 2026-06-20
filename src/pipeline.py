from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import cv2

from src.counting import EntryCounter
from src.detector import YOLODetector
from src.tracker import DeepSortTracker
from src.visualization import draw_annotations


@dataclass
class PipelineConfig:
    model_path: str = "yolov8n.pt"
    confidence: float = 0.35
    iou: float = 0.45
    imgsz: int = 640
    device: str | None = None
    use_gpu: bool = False
    count_enabled: bool = True
    count_line_ratio: float = 0.5


@dataclass
class FPSMeter:
    last_time: float = field(default_factory=time.perf_counter)
    fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        delta = now - self.last_time
        self.last_time = now
        if delta > 0:
            current = 1.0 / delta
            self.fps = current if self.fps == 0.0 else self.fps * 0.9 + current * 0.1
        return self.fps


class ObjectTrackingPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.detector = YOLODetector(
            model_path=config.model_path,
            confidence=config.confidence,
            iou=config.iou,
            imgsz=config.imgsz,
            device=config.device,
        )
        self.tracker = DeepSortTracker(use_gpu=config.use_gpu)
        self.entry_counter = EntryCounter(line_ratio=config.count_line_ratio)
        self.fps_meter = FPSMeter()
        self.selected_class_names: list[str] = []
        self.active_track_count = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.entry_counter.reset()
        self.fps_meter = FPSMeter()
        self.active_track_count = 0

    def _selected_class_ids(self) -> list[int] | None:
        if not self.selected_class_names:
            return None
        class_map = {name: class_id for class_id, name in self.detector.class_names.items()}
        return [class_map[name] for name in self.selected_class_names if name in class_map]

    def process_frame(self, frame):
        class_ids = self._selected_class_ids()
        detections = self.detector.detect(frame, class_ids=class_ids)
        tracked_objects = self.tracker.update(frame, detections)

        fps = self.fps_meter.tick()
        self.active_track_count = len(tracked_objects)

        if self.config.count_enabled:
            for obj in tracked_objects:
                self.entry_counter.update(obj.track_id, obj.centroid, frame.shape[0], obj.class_name)

        annotated = draw_annotations(
            frame,
            tracked_objects,
            count_value=self.entry_counter.total_count,
            fps=fps,
            line_ratio=self.config.count_line_ratio,
        )

        summary = {
            "fps": fps,
            "detections": len(detections),
            "tracks": len(tracked_objects),
            "people_counted": self.entry_counter.total_count,
        }
        return annotated, summary


def process_video_file(
    input_path: str,
    output_path: str | None,
    pipeline: ObjectTrackingPipeline,
    progress_callback=None,
):
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )

    pipeline.reset()
    frame_index = 0
    last_frame = None
    start_time = time.perf_counter()

    while True:
        success, frame = capture.read()
        if not success:
            break

        annotated, _ = pipeline.process_frame(frame)
        last_frame = annotated
        if writer is not None:
            writer.write(annotated)

        frame_index += 1
        if progress_callback is not None:
            progress_callback(frame_index, total_frames)

    capture.release()
    if writer is not None:
        writer.release()

    average_fps = 0.0
    elapsed = time.perf_counter() - start_time
    if frame_index > 0 and elapsed > 0:
        average_fps = frame_index / elapsed

    return {
        "frames_processed": frame_index,
        "average_fps": average_fps,
        "people_counted": pipeline.entry_counter.total_count,
        "last_frame": last_frame,
    }
