from typing import Annotated

from fastapi import Header, HTTPException, Request, UploadFile, status

from app.config import settings
from app.model import ImageClassifier


def get_classifier(request: Request) -> ImageClassifier:
    """Retrieves the ImageClassifier instance from the application state.

    Args:
        request: The incoming FastAPI request object containing the app state.

    Returns:
        The globally initialized ImageClassifier instance.
    """
    return request.app.state.classifier


async def validate_image(
    image: UploadFile, content_length: Annotated[int | None, Header()] = None
):
    """Validates the uploaded image file for type and size constraints.

    This dependency checks the MIME type to ensure the file is an image and
    enforces file size limits using both the Content-Length header and
    the actual byte count of the file content.

    Args:
        image: The uploaded file object to be validated.
        content_length: Optional Content-Length header for early size validation.

    Returns:
        The raw bytes content of the validated image.

    Raises:
        HTTPException: 400 Bad Request if the uploaded file is not an image.
        HTTPException: 413 Request Entity Too Large if the file size exceeds
            the limit defined in settings.
    """
    max_bytes = settings.max_file_size
    max_mb = max_bytes // (1024 * 1024)

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image"
        )

    if content_length and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (header). Max: {max_mb}MB",
        )

    content = await image.read()

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (actual). Max: {max_mb}MB",
        )

    return content
