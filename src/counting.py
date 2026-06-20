from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntryCounter:
    line_ratio: float = 0.5
    count_class_name: str = "person"
    total_count: int = 0
    counted_track_ids: set[str] = field(default_factory=set)
    previous_centroids: dict[str, tuple[float, float]] = field(default_factory=dict)

    def reset(self) -> None:
        self.total_count = 0
        self.counted_track_ids.clear()
        self.previous_centroids.clear()

    def update(self, track_id: str, centroid: tuple[float, float], frame_height: int, class_name: str | None) -> bool:
        if class_name is None or class_name != self.count_class_name:
            self.previous_centroids[track_id] = centroid
            return False

        line_y = frame_height * self.line_ratio
        previous = self.previous_centroids.get(track_id)
        self.previous_centroids[track_id] = centroid

        if previous is None or track_id in self.counted_track_ids:
            return False

        crossed_downward = previous[1] < line_y <= centroid[1]
        if crossed_downward:
            self.total_count += 1
            self.counted_track_ids.add(track_id)
            return True
        return False
