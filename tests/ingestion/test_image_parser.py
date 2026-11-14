"""Tests for multimodal image parser."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.ingestion.image_parser import (
    parse_image,
    IMAGE_EXTENSIONS,
    _encode_image_base64,
    _get_mime_type,
    VISION_EXTRACTION_PROMPT,
)
from lattice.ingestion.models import DocumentType


class TestImageParserBasics:

    def test_image_extensions(self):
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS
        assert ".txt" not in IMAGE_EXTENSIONS

    def test_mime_types(self, tmp_path):
        assert _get_mime_type(Path("photo.png")) == "image/png"
        assert _get_mime_type(Path("photo.jpg")) == "image/jpeg"
        assert _get_mime_type(Path("photo.jpeg")) == "image/jpeg"
        assert _get_mime_type(Path("photo.webp")) == "image/webp"
        assert _get_mime_type(Path("photo.gif")) == "image/gif"

    def test_encode_base64(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        result = _encode_image_base64(f)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extraction_prompt_exists(self):
        assert "entities" in VISION_EXTRACTION_PROMPT.lower()
        assert "relationships" in VISION_EXTRACTION_PROMPT.lower()


class TestParseImageNoProvider:

    def test_no_provider_returns_placeholder(self, tmp_path):
        f = tmp_path / "diagram.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        doc = parse_image(f, llm_provider=None)

        assert doc.metadata.doc_type == DocumentType.IMAGE
        assert doc.metadata.title == "diagram"
        assert "No vision-capable LLM" in doc.content
        assert doc.metadata.extra["mime_type"] == "image/png"

    def test_image_metadata(self, tmp_path):
        f = tmp_path / "photo.jpg"
        content = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        f.write_bytes(content)

        doc = parse_image(f, llm_provider=None)

        assert doc.metadata.extra["file_size"] == len(content)
        assert doc.metadata.extra["mime_type"] == "image/jpeg"
        assert doc.metadata.source == str(f)


class TestParseImageWithMockProvider:

    def test_vision_extraction_with_mock_ollama(self, tmp_path):
        f = tmp_path / "arch.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        mock_response_text = """## Summary
An architecture diagram showing a web application.

## Entities
ENTITY: APIGateway (service)
ENTITY: Database (storage)
ENTITY: AuthService (service)

## Relationships
RELATION: APIGateway -> Database (reads_from)
RELATION: APIGateway -> AuthService (authenticates_via)
"""
        mock_provider = MagicMock()
        mock_provider._base_url = "http://localhost:11434"
        mock_provider._model = "llava"

        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = {
            "message": {"content": mock_response_text}
        }
        mock_response_obj.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response_obj
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            doc = parse_image(f, llm_provider=mock_provider)

        assert "APIGateway" in doc.content
        assert "Database" in doc.content
        assert doc.metadata.doc_type == DocumentType.IMAGE

    def test_vision_failure_fallback(self, tmp_path):
        f = tmp_path / "bad.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 10)

        mock_provider = MagicMock()
        mock_provider._base_url = "http://localhost:11434"
        mock_provider._model = "llava"

        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            doc = parse_image(f, llm_provider=mock_provider)

        assert doc.metadata.doc_type == DocumentType.IMAGE
        assert isinstance(doc.content, str)
