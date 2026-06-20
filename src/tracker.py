from __future__ import annotations

from dataclasses import dataclass

from deep_sort_realtime.deepsort_tracker import DeepSort


@dataclass(frozen=True)
class TrackedObject:
    track_id: str
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str | None
    confidence: float | None

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class DeepSortTracker:
    def __init__(self, use_gpu: bool = False) -> None:
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            nms_max_overlap=1.0,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=use_gpu,
        )

    def reset(self) -> None:
        self.tracker.delete_all_tracks()

    def update(self, frame, detections) -> list[TrackedObject]:
        raw_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            raw_detections.append(
                ([x1, y1, x2 - x1, y2 - y1], detection.confidence, detection.class_name)
            )

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        tracked_objects: list[TrackedObject] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb(orig=True)
            if bbox is None:
                bbox = track.to_ltrb()
            if bbox is None:
                continue

            tracked_objects.append(
                TrackedObject(
                    track_id=str(track.track_id),
                    bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    class_name=track.get_det_class(),
                    confidence=track.get_det_conf(),
                )
            )

        return tracked_objects
