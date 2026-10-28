from pathlib import Path
import io
import fitz
import pdfplumber
from PIL import Image

from src.config import IMAGE_DIR


class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(self.pdf_path)

    def extract_text(self):
        items = []
        with fitz.open(self.pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    items.append({
                        "id": f"text-page-{page_number}",
                        "modality": "text",
                        "page": page_number,
                        "content": text,
                        "metadata": {"source": self.pdf_path.name},
                    })
        return items

    def extract_images(self):
        items = []
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        with fitz.open(self.pdf_path) as doc:
            counter = 0
            for page_number, page in enumerate(doc, start=1):
                for image_info in page.get_images(full=True):
                    xref = image_info[0]
                    base = doc.extract_image(xref)
                    image_bytes = base["image"]
                    ext = base["ext"]

                    try:
                        Image.open(io.BytesIO(image_bytes)).verify()
                    except Exception:
                        continue

                    counter += 1
                    path = IMAGE_DIR / f"page_{page_number}_image_{counter}.{ext}"
                    path.write_bytes(image_bytes)

                    items.append({
                        "id": f"image-{counter}",
                        "modality": "image",
                        "page": page_number,
                        "image_bytes": image_bytes,
                        "image_path": str(path),
                        "metadata": {
                            "source": self.pdf_path.name,
                            "format": ext,
                        },
                    })
        return items

    def extract_tables(self):
        items = []
        with pdfplumber.open(self.pdf_path) as pdf:
            counter = 0
            for page_number, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables():
                    if not table:
                        continue
                    cleaned = [
                        row for row in table
                        if row and any(cell not in (None, "") for cell in row)
                    ]
                    if not cleaned:
                        continue

                    counter += 1
                    items.append({
                        "id": f"table-{counter}",
                        "modality": "table",
                        "page": page_number,
                        "content": cleaned,
                        "metadata": {"source": self.pdf_path.name},
                    })
        return items

    def extract(self):
        return {
            "text": self.extract_text(),
            "images": self.extract_images(),
            "tables": self.extract_tables(),
        }
