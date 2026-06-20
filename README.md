# Real-Time AI-Powered Surveillance System

An industry-grade, full-stack web application for real-time object detection and multi-object tracking. Built using **YOLOv8**, **Deep SORT**, **FastAPI**, and **React**.

## Features
- **Object Detection & Tracking:** GPU-accelerated YOLOv8 + Deep SORT.
- **Asynchronous Processing:** Celery-backed processing for smooth video processing.
- **Modern Dashboard:** React (Vite) + TailwindCSS.
- **Advanced Capabilities:** Line-crossing object counting, FPS monitoring, class filtering.
- **Dockerized:** Simple `docker-compose up` setup.

## Quick Start
```bash
docker-compose up --build
```
- Frontend: `http://localhost:3000`
- Backend API Docs: `http://localhost:8000/docs`

## Architecture
- **Backend:** FastAPI, Celery, Redis, SQLAlchemy.
- **AI Core:** Ultralytics (YOLOv8), Deep SORT, OpenCV, PyTorch.
- **Frontend:** React, Vite, Tailwind CSS.

## License
MIT License
