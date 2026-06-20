from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logger import logger

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for media (processed videos, heatmaps)
    app.mount("/static", StaticFiles(directory=settings.MEDIA_ROOT), name="static")

    @app.get("/")
    def root():
        return {"message": "Welcome to the Object Detection & Tracking API"}

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    from app.api.v1.endpoints import video
    app.include_router(video.router, prefix=settings.API_V1_STR)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
