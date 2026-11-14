"""
Multimodal image parser using LLM vision capabilities.

Sends images to a vision-capable LLM to extract text, entities, and
relationships from diagrams, screenshots, whiteboard photos, etc.
Falls back gracefully if no vision provider is configured.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import structlog

from lattice.ingestion.models import Document, DocumentMetadata, DocumentType

logger = structlog.get_logger()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

VISION_EXTRACTION_PROMPT = """\
Analyze this image and extract structured information. Provide:

1. **Summary**: A concise description of what the image shows.
2. **Visible Text**: Any text visible in the image, transcribed exactly.
3. **Entities**: Named entities found (people, systems, components, services, concepts). List each on its own line as: ENTITY: <name> (<type>)
4. **Relationships**: Connections between entities. List each as: RELATION: <source> -> <target> (<relationship_type>)

Be thorough but concise. If the image is a diagram, capture the full structure.
If it's a screenshot, focus on the key content and UI elements.
"""


def _encode_image_base64(path: Path) -> str:
    """Read an image file and return its base64-encoded content."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _get_mime_type(path: Path) -> str:
    """Map file extension to MIME type."""
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")


async def _call_vision_llm(path: Path, provider) -> Optional[str]:
    """Send image to an LLM with vision support and return extracted text."""
    try:
        import httpx

        b64 = _encode_image_base64(path)
        mime = _get_mime_type(path)

        if hasattr(provider, "_base_url"):
            payload = {
                "model": provider._model,
                "messages": [
                    {
                        "role": "user",
                        "content": VISION_EXTRACTION_PROMPT,
                        "images": [b64],
                    }
                ],
                "stream": False,
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{provider._base_url}/api/chat", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
            return data.get("message", {}).get("content", "")

        if hasattr(provider, "_async_client"):
            from lattice.inference.provider import Message, GenerationConfig

            msg = Message(
                role="user",
                content=f"[Image: {path.name} ({mime})]\n\n{VISION_EXTRACTION_PROMPT}\n\n"
                        f"(Image data is base64 encoded, {len(b64)} chars)",
            )
            result = await provider.generate(
                messages=[msg],
                config=GenerationConfig(temperature=0.3, max_tokens=2048),
            )
            return result.content

    except Exception as e:
        logger.warning("vision_llm_call_failed", path=str(path), error=str(e))
    return None


def parse_image(path: Path, llm_provider=None) -> Document:
    """Parse an image file by extracting content via LLM vision.

    If no LLM provider is available, creates a minimal Document
    with just the file metadata so the pipeline doesn't break.
    """
    path = Path(path)

    content = ""

    if llm_provider is not None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                content = pool.submit(
                    asyncio.run, _call_vision_llm(path, llm_provider)
                ).result() or ""
        else:
            content = asyncio.run(_call_vision_llm(path, llm_provider)) or ""
    else:
        logger.warning(
            "no_vision_provider",
            path=str(path),
            msg="Skipping image content extraction — no LLM provider configured",
        )
        content = f"[Image file: {path.name}]\n\nNo vision-capable LLM configured for content extraction."

    metadata = DocumentMetadata(
        source=str(path),
        title=path.stem,
        doc_type=DocumentType.IMAGE,
        extra={"mime_type": _get_mime_type(path), "file_size": path.stat().st_size},
    )

    logger.info("image_parsed", path=str(path), content_chars=len(content))
    return Document(content=content, metadata=metadata)
