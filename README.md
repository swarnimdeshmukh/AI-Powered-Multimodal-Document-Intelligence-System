# Multi-Modal RAG System — Local / No-API

A local, portfolio-grade Multi-Modal Retrieval-Augmented Generation system for PDFs containing **text, images/charts, and tables**.

## Architecture

PDF → extraction → text/images/tables → local multimodal embeddings → ChromaDB → multimodal retrieval → local vision-language model → answer + source pages.

This project intentionally uses **no OpenAI, Anthropic, Gemini, Claude API, or paid cloud inference**.

## Stack

- Python 3.11
- PyMuPDF
- pdfplumber
- Pillow
- Pandas
- PyTorch
- OpenCLIP
- ChromaDB
- Streamlit
- Local vision-language model (added/configured in the generation phase)

## Project structure

```text
multimodal-rag-system/
├── data/
│   └── annual_report.pdf
├── storage/
│   ├── chroma/
│   └── extracted_images/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py
│   │   ├── image_processor.py
│   │   └── table_processor.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── multimodal_embedder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   └── generation/
│       ├── __init__.py
│       └── local_vlm.py
├── tests/
│   ├── __init__.py
│   ├── test_extraction.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
├── evaluation/
│   ├── questions.json
│   └── evaluate.py
├── app.py
├── test_extraction.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
conda create -n multimodal-rag python=3.11 -y
conda activate multimodal-rag
pip install -r requirements.txt
```

Put a PDF containing text, charts/images, and tables into `data/` as `annual_report.pdf`.

Then run:

```bash
python test_extraction.py
```

We will implement the remaining stages incrementally and test each stage before moving forward.

## No-API rule

Do not add API keys or cloud inference. Models used for embeddings and final generation are intended to run locally.
