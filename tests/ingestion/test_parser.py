"""PDF parsing on the installed pypdf.

A truncated cross-reference table with no trailer used to make pypdf loop forever
(GHSA-5849-834g-hvw5). parse_pdf must finish instead of hanging.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path

from lattice.ingestion.models import DocumentType
from lattice.ingestion.parser import parse_pdf

# Standard xref that runs into EOF without a trailer keyword.
# pypdf < 6.21.4 rewinds to the same offset and never returns.
_TRUNCATED_XREF_NO_TRAILER = b"%PDF-1.4\nstartxref\n28\n%%EOF\nxref\n1 0\naaa"


class _ParseHung(BaseException):
    """Raised by the alarm so a parser `except Exception` cannot swallow the timeout."""


def _minimal_text_pdf(text: str) -> bytes:
    """A one-page PDF whose content stream draws `text` with Helvetica."""
    stream = f"BT /F1 24 Tf 72 120 Td ({text}) Tj ET\n".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_at = len(body)
    body += f"xref\n0 {len(objects) + 1}\n".encode()
    body += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        body += f"{offset:010d} 00000 n \n".encode()
    body += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(body)


def test_truncated_pdf_without_trailer_finishes_within_five_seconds(tmp_path: Path) -> None:
    pdf = tmp_path / "truncated-no-trailer.pdf"
    pdf.write_bytes(_TRUNCATED_XREF_NO_TRAILER)

    def _on_timeout(signum: int, frame: object) -> None:
        raise _ParseHung("parse_pdf did not finish within 5 seconds")

    previous = signal.signal(signal.SIGALRM, _on_timeout)
    signal.setitimer(signal.ITIMER_REAL, 5)
    started = time.monotonic()
    finished = False
    try:
        try:
            parse_pdf(pdf)
            finished = True
        except _ParseHung:
            raise
        except Exception:
            # Raising on this malformed file is the patched pypdf behavior.
            finished = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

    assert finished
    assert time.monotonic() - started < 5


def test_parse_pdf_extracts_text_from_a_valid_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "hello.pdf"
    pdf.write_bytes(_minimal_text_pdf("Hello Lattice"))

    doc = parse_pdf(pdf)

    assert doc.content == "Hello Lattice"
    assert doc.metadata.doc_type == DocumentType.PDF
    assert doc.metadata.source == str(pdf)
    assert doc.metadata.title == "hello"
