from pathlib import Path

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def inspect_image(file_path: str) -> dict:
    """
    Inspect an image and return its basic properties.
    """

    path = Path(file_path)

    if not path.exists():
        return {
            "error": f"Image file not found: {file_path}"
        }

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return {
            "error": (
                "Unsupported image format. "
                "Use PNG, JPG, JPEG, or WEBP."
            )
        }

    try:
        with Image.open(path) as image:
            return {
                "file_name": path.name,
                "format": image.format,
                "width": int(image.width),
                "height": int(image.height),
                "mode": image.mode,
                "channels": len(image.getbands()),
                "aspect_ratio": round(
                    image.width / image.height,
                    4,
                ),
                "file_size_bytes": path.stat().st_size,
            }

    except Exception as exc:
        return {
            "error": f"Unable to inspect image: {exc}"
        }