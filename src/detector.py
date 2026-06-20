from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.class_names = dict(self.model.names)

    def set_model(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        self.class_names = dict(self.model.names)

    def detect(self, frame, class_ids: list[int] | None = None) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=class_ids,
            verbose=False,
        )

        if not results:
            return []

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for xyxy, conf, cls_id in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            class_index = int(cls_id)
            detections.append(
                Detection(
                    bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=float(conf),
                    class_id=class_index,
                    class_name=self.class_names.get(class_index, str(class_index)),
                )
            )

        return detections
