from __future__ import annotations

import tempfile
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from src.detector import YOLODetector
from src.pipeline import ObjectTrackingPipeline, PipelineConfig, process_video_file


st.set_page_config(page_title="Object Detection + Tracking", page_icon="🎥", layout="wide")


class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self, pipeline: ObjectTrackingPipeline) -> None:
        self.pipeline = pipeline

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        annotated, _ = self.pipeline.process_frame(image)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


@st.cache_resource(show_spinner=False)
def _load_class_names(model_path: str) -> list[str]:
    detector = YOLODetector(model_path=model_path)
    return list(detector.class_names.values())


def _pipeline_cache_key(config: PipelineConfig) -> str:
    return "|".join(
        [
            config.model_path,
            str(config.confidence),
            str(config.iou),
            str(config.imgsz),
            str(config.device),
            str(config.count_enabled),
            str(config.count_line_ratio),
        ]
    )


def _get_or_create_pipeline(config: PipelineConfig) -> ObjectTrackingPipeline:
    cache_key = _pipeline_cache_key(config)
    if "pipelines" not in st.session_state:
        st.session_state.pipelines = {}
    if cache_key not in st.session_state.pipelines:
        st.session_state.pipelines[cache_key] = ObjectTrackingPipeline(config)
    return st.session_state.pipelines[cache_key]


def _build_pipeline_from_sidebar() -> ObjectTrackingPipeline:
    model_path = st.sidebar.text_input(
        "YOLO model",
        value="yolov8n.pt",
    )
    confidence = st.sidebar.slider(
        "Confidence threshold",
        0.05,
        0.95,
        0.35,
        0.05,
    )
    iou = st.sidebar.slider(
        "NMS IoU threshold",
        0.1,
        0.95,
        0.45,
        0.05,
    )
    imgsz = st.sidebar.select_slider(
        "Inference size",
        options=[320, 416, 512, 640, 768, 960],
        value=640,
    )
    device = st.sidebar.selectbox(
        "Device",
        ["", "cpu", "cuda"],
        index=0,
    )
    count_enabled = st.sidebar.checkbox(
        "Enable counting line",
        value=True,
    )
    count_ratio = st.sidebar.slider(
        "Counting line position",
        0.1,
        0.9,
        0.5,
        0.05,
    )
    class_names = _load_class_names(model_path)
    default_classes = ["person"] if "person" in class_names else class_names[:3]
    selected_class_names = st.sidebar.multiselect(
        "Detect only these classes",
        options=class_names,
        default=default_classes,
    )

    config = PipelineConfig(
        model_path=model_path,
        confidence=confidence,
        iou=iou,
        imgsz=imgsz,
        device=device or None,
        count_enabled=count_enabled,
        count_line_ratio=count_ratio,
        use_gpu=device == "cuda",
    )
    pipeline = _get_or_create_pipeline(config)
    pipeline.selected_class_names = selected_class_names
    return pipeline


def _sidebar_status(pipeline: ObjectTrackingPipeline) -> None:
    st.sidebar.markdown("### Current classes")
    if pipeline.selected_class_names:
        st.sidebar.write(", ".join(pipeline.selected_class_names))
    else:
        st.sidebar.write("All classes")

    st.sidebar.markdown("### Tracking")
    st.sidebar.write(f"Active tracks: {pipeline.active_track_count}")
    st.sidebar.write(f"People counted: {pipeline.entry_counter.total_count}")
    st.sidebar.write(f"Average FPS: {pipeline.fps_meter.fps:.2f}")


def main() -> None:
    st.title("Advanced Real-Time Object Detection and Tracking")
    st.write(
        "YOLOv8 detects objects, Deep SORT keeps IDs stable across frames, and the app can count line crossings in real time."
    )

    with st.sidebar:
        st.header("Controls")
        st.caption("Choose your model and runtime settings, then run either live webcam or video file mode.")
        pipeline = _build_pipeline_from_sidebar()
        _sidebar_status(pipeline)

    tab_live, tab_file = st.tabs(["Live Webcam", "Video File"])

    with tab_live:
        st.subheader("Live webcam tracking")
        webrtc_streamer(
            key="yolo-deepsort-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: LiveVideoProcessor(pipeline),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.info("The live mode uses your browser camera and processes each frame with the shared YOLO + Deep SORT pipeline.")

    with tab_file:
        st.subheader("Upload a video for offline processing")

        uploaded = st.file_uploader("Upload MP4, MOV, or AVI", type=["mp4", "mov", "avi", "mkv"])
        save_output = st.checkbox("Save annotated output video", value=True)

        if uploaded is not None:
            temp_input = Path(tempfile.gettempdir()) / uploaded.name
            temp_input.write_bytes(uploaded.getbuffer())

            output_dir = Path(tempfile.gettempdir()) / "object_tracking_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"annotated_{uploaded.name.rsplit('.', 1)[0]}.mp4"

            if st.button("Process video", type="primary"):
                progress = st.progress(0)
                status = st.empty()

                def update_progress(frame_index: int, total_frames: int) -> None:
                    if total_frames > 0:
                        progress.progress(min(frame_index / total_frames, 1.0))
                    status.write(f"Processing frame {frame_index}/{total_frames}")

                summary = process_video_file(
                    input_path=str(temp_input),
                    output_path=str(output_path) if save_output else None,
                    pipeline=pipeline,
                    progress_callback=update_progress,
                )

                progress.progress(1.0)
                status.success("Processing complete")

                st.success(
                    f"Processed {summary['frames_processed']} frames at {summary['average_fps']:.2f} FPS. "
                    f"People counted: {summary['people_counted']}"
                )

                if save_output and output_path.exists():
                    with open(output_path, "rb") as file_handle:
                        st.download_button(
                            label="Download annotated video",
                            data=file_handle,
                            file_name=output_path.name,
                            mime="video/mp4",
                        )

                preview_frame = summary.get("last_frame")
                if preview_frame is not None:
                    st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), caption="Last annotated frame")
        else:
            st.info("Upload a video to run YOLOv8 + Deep SORT on every frame and save the annotated result.")


if __name__ == "__main__":
    main()
