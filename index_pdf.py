from pathlib import Path
import shutil
import fitz

from src.ingestion.pdf_extractor import PDFExtractor
from src.embeddings.multimodal_embedder import MultiModalEmbedder
from src.retrieval.vector_store import VectorStore


CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

PAGE_IMAGE_DIR = Path(
    "storage/page_images"
)


# =============================================================
# TEXT CHUNKING
# =============================================================

def chunk_text(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(
                chunk
            )

        start += (
            CHUNK_SIZE -
            CHUNK_OVERLAP
        )

    return chunks


# =============================================================
# RENDER PDF PAGES
# =============================================================

def render_pdf_pages(
    pdf_path,
    output_dir=PAGE_IMAGE_DIR
):

    output_dir = Path(
        output_dir
    )

    # Remove old page images
    if output_dir.exists():

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    doc = fitz.open(
        str(pdf_path)
    )

    page_paths = []

    print(
        "\nRendering PDF pages..."
    )

    total_pages = len(doc)

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        output_path = (
            output_dir /
            f"page_{page_number}.png"
        )

        matrix = fitz.Matrix(
            2,
            2
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        pixmap.save(
            str(output_path)
        )

        page_paths.append(
            str(output_path)
        )

        print(
            f"  ✓ Page "
            f"{page_number}/{total_pages}"
        )

    doc.close()

    return page_paths


# =============================================================
# INDEX PDF
# =============================================================

def index_pdf(
    pdf_path,
    reset=True
):

    pdf_path = Path(
        pdf_path
    )

    print("=" * 60)
    print("MULTI-MODAL RAG INDEXING")
    print("=" * 60)

    print(
        f"\nDocument: {pdf_path.name}"
    )

    # ---------------------------------------------------------
    # 1. EXTRACT PDF
    # ---------------------------------------------------------

    print(
        "\n[1/5] Extracting PDF..."
    )

    extractor = PDFExtractor(
        str(pdf_path)
    )

    result = extractor.extract()

    texts = result.get(
        "text",
        []
    )

    images = result.get(
        "images",
        []
    )

    tables = result.get(
        "tables",
        []
    )

    print(
        f"Text blocks : {len(texts)}"
    )

    print(
        f"Images      : {len(images)}"
    )

    print(
        f"Tables      : {len(tables)}"
    )

    # ---------------------------------------------------------
    # 2. RENDER FULL PAGES
    # ---------------------------------------------------------

    print(
        "\n[2/5] Creating page-level images..."
    )

    page_paths = render_pdf_pages(
        pdf_path
    )

    print(
        f"\nComplete page images: "
        f"{len(page_paths)}"
    )

    # ---------------------------------------------------------
    # 3. LOAD EMBEDDER
    # ---------------------------------------------------------

    print(
        "\n[3/5] Loading embedding models..."
    )

    embedder = MultiModalEmbedder()

    # ---------------------------------------------------------
    # 4. VECTOR STORE
    # ---------------------------------------------------------

    print(
        "\n[4/5] Connecting to ChromaDB..."
    )

    store = VectorStore()

    if reset:

        store.clear()

    # ---------------------------------------------------------
    # 5. INDEX
    # ---------------------------------------------------------

    print(
        "\n[5/5] Indexing content..."
    )

    text_count = 0
    table_count = 0
    page_count = 0

    # =========================================================
    # TEXT
    # =========================================================

    print(
        "\nIndexing text chunks..."
    )

    for page_index, item in enumerate(
        texts
    ):

        text = item.get(
            "content",
            ""
        )

        if not text.strip():

            continue

        page = item.get(
            "page",
            ""
        )

        chunks = chunk_text(
            text
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            embedding = embedder.embed_text(
                chunk
            )

            store.add_text(

                embedding=embedding,

                document=chunk,

                metadata={
                    "type": "text",
                    "source": str(
                        pdf_path
                    ),
                    "page": str(page),
                    "chunk": str(
                        chunk_index
                    ),
                },

                item_id=(
                    f"text-"
                    f"{page_index}-"
                    f"{chunk_index}"
                ),
            )

            text_count += 1

        print(
            f"  ✓ Page {page} → "
            f"{len(chunks)} chunks"
        )

    # =========================================================
    # TABLES
    # =========================================================

    print(
        "\nIndexing tables..."
    )

    for i, item in enumerate(
        tables
    ):

        table_content = item.get(
            "content",
            []
        )

        if not isinstance(
            table_content,
            list
        ):

            continue

        rows = []

        for row in table_content:

            if isinstance(
                row,
                (list, tuple)
            ):

                rows.append(
                    " | ".join(
                        str(cell)
                        for cell in row
                    )
                )

        table_text = "\n".join(
            rows
        ).strip()

        if len(table_text) < 15:

            continue

        embedding = embedder.embed_text(
            table_text
        )

        store.add_text(

            embedding=embedding,

            document=table_text,

            metadata={
                "type": "table",
                "source": str(
                    pdf_path
                ),
                "page": str(
                    item.get(
                        "page",
                        ""
                    )
                ),
            },

            item_id=f"table-{i}",
        )

        table_count += 1

        print(
            f"  ✓ Table "
            f"{table_count}"
        )

    # =========================================================
    # FULL PAGE IMAGES
    # =========================================================

    print(
        "\nIndexing complete page images..."
    )

    for page_number, image_path in enumerate(
        page_paths,
        start=1
    ):

        embedding = embedder.embed_image(
            image_path
        )

        store.add_image(

            embedding=embedding,

            document=(
                f"Complete PDF page "
                f"{page_number}"
            ),

            metadata={
                "type": "page_image",
                "source": image_path,
                "page": str(
                    page_number
                ),
            },

            item_id=(
                f"page-image-"
                f"{page_number}"
            ),
        )

        page_count += 1

        print(
            f"  ✓ Page image "
            f"{page_number}/"
            f"{len(page_paths)}"
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "INDEXING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Text chunks       : "
        f"{text_count}"
    )

    print(
        f"Tables            : "
        f"{table_count}"
    )

    print(
        f"Page images       : "
        f"{page_count}"
    )

    print(
        f"\nText collection   : "
        f"{store.text_count()}"
    )

    print(
        f"Image collection  : "
        f"{store.image_count()}"
    )

    print(
        "=" * 60
    )

    return {
        "pdf": str(pdf_path),
        "filename": pdf_path.name,
        "text_chunks": text_count,
        "tables": table_count,
        "page_images": page_count,
        "text_collection": store.text_count(),
        "image_collection": store.image_count(),
    }


# =============================================================
# COMMAND LINE
# =============================================================

def main():

    pdf_path = Path(
        "data/annual_report.pdf"
    )

    index_pdf(
        pdf_path,
        reset=True
    )


if __name__ == "__main__":

    main()