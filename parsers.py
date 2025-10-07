"""
parsers.py
----------
Two interchangeable parsers so you can A/B complex resume parsing:
- BaselineParser: pdfplumber/docx2txt + OCR fallback via Tesseract
- DoclingParser : IBM Docling (layout-aware; exports Markdown)

Both expose a common .parse(file_bytes, file_ext) -> str interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal
import io, os, tempfile

# Baseline stack
import pdfplumber, docx2txt, pytesseract
from pdf2image import convert_from_bytes

# Docling
from docling.document_converter import DocumentConverter

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes, file_ext: str) -> str: ...

class BaselineParser(BaseParser):
    """Hybrid text-layer + OCR resume parser."""
    def parse(self, file_bytes: bytes, file_ext: str) -> str:
        ext = file_ext.lower()
        if ext == ".pdf":
            text = ""
            # Try text-layer extraction
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except Exception:
                text = ""
            # OCR fallback for images/complex layouts
            if not text.strip():
                images = convert_from_bytes(file_bytes)
                for img in images:
                    text += pytesseract.image_to_string(img)
            return text.strip()

        if ext == ".docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_bytes); path = tmp.name
            try:
                return (docx2txt.process(path) or "").strip()
            finally:
                os.remove(path)

        raise ValueError("Unsupported file type for BaselineParser (use .pdf or .docx)")

class DoclingParser(BaseParser):
    """Layout-aware parsing using Docling; exports Markdown preserving structure."""
    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_bytes: bytes, file_ext: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_bytes); path = tmp.name
        try:
            dl_doc = self.converter.convert(path).document
            md = dl_doc.export_to_markdown()
            return md.strip()
        finally:
            os.remove(path)

def get_parser(backend: Literal["baseline","docling"]="baseline") -> BaseParser:
    return DoclingParser() if backend == "docling" else BaselineParser()