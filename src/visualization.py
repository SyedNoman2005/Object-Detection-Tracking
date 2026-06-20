from __future__ import annotations

import colorsys
import hashlib

import cv2


def _color_from_id(track_id: str) -> tuple[int, int, int]:
    digest = hashlib.md5(track_id.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16) / 255.0
    saturation = 0.85
    value = 0.95
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_annotations(frame, tracked_objects, count_value: int = 0, fps: float = 0.0, line_ratio: float = 0.5):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    line_y = int(height * line_ratio)
    cv2.line(annotated, (0, line_y), (width, line_y), (0, 255, 255), 2)

    for obj in tracked_objects:
        x1, y1, x2, y2 = [int(value) for value in obj.bbox_xyxy]
        color = _color_from_id(obj.track_id)
        label = obj.class_name or "object"
        confidence_text = f"{obj.confidence:.2f}" if obj.confidence is not None else "?"
        caption = f"ID {obj.track_id} | {label} {confidence_text}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top_left_y = max(y1 - text_size[1] - baseline - 8, 0)
        cv2.rectangle(
            annotated,
            (x1, top_left_y),
            (x1 + text_size[0] + 10, top_left_y + text_size[1] + baseline + 8),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            caption,
            (x1 + 5, top_left_y + text_size[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    overlay_text = f"Count: {count_value} | FPS: {fps:.2f}"
    cv2.rectangle(annotated, (10, 10), (310, 52), (20, 20, 20), -1)
    cv2.putText(annotated, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated
