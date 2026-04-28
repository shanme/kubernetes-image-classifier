import asyncio
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator, metrics

from app.config import settings
from app.dependencies import get_classifier, validate_image
from app.model import ImageClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifecycle.

    Initializes the ImageClassifier during startup and attaches it to the
    application state. Ensures the classifier is removed from memory on shutdown.
    """
    torch.set_num_threads(1)
    app.state.inference_semaphore = asyncio.Semaphore(1)
    app.state.classifier = ImageClassifier(settings)

    yield

    del app.state.classifier
    del app.state.inference_semaphore


app = FastAPI(lifespan=lifespan)

instrumentator = Instrumentator()
instrumentator.add(
    metrics.default(
        latency_lowr_buckets=(0.1, 0.5, 1.0, 1.5, 2, 2.5, 3, 5, 7.5, 10, 30, 60)
    )
)
instrumentator.instrument(app).expose(app)


@app.middleware("http")
async def force_close_connection(request: Request, call_next):
    """Forces the closure of the HTTP connection after each request.

    Sets the 'Connection: close' header in the response to prevent
    persistent (keep-alive) connections.
    """
    response = await call_next(request)
    response.headers["Connection"] = "close"
    return response


@app.post("/predict")
async def handle_predict(
    request: Request,
    image_bytes: bytes = Depends(validate_image),
    classifier: ImageClassifier = Depends(get_classifier),
):
    """Processes an image upload and returns classification results.

    This endpoint receives validated image bytes, performs inference using
    the PyTorch model, and returns labels with confidence scores.
    Inference is executed in a separate threadpool to prevent blocking
    the main event loop during heavy ML computations.

    Args:
        request: Request used to access app.state
        image_bytes: Raw bytes of the validated image file.
        classifier: The initialized ImageClassifier instance.

    Returns:
        A dictionary containing the predicted class label and confidence level.

    Raises:
        HTTPException: 400 Bad Request if the image file is corrupted or unreadable.
        HTTPException: 500 Internal Server Error if an unexpected error occurs
            during inference.
    """
    try:
        async with request.app.state.inference_semaphore:
            return await run_in_threadpool(classifier.predict, image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file"
        ) from None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None


@app.get("/health", tags=["System"])
async def health_check():
    """Check the service availability.

    Used by Kubernetes liveness and readiness probes to ensure the
    container is operational.

    Returns:
        A dictionary with the current status of the service.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)