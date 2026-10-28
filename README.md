# IntelliRAG — AI-Powered Multimodal Document Intelligence

IntelliRAG is a multimodal Retrieval-Augmented Generation (RAG) system designed to enable intelligent document understanding through semantic retrieval, contextual question answering, and document analysis.

## Key Features

- **Multimodal document processing:** Extract and process PDF text, images, and tables.
- **Context-aware question answering:** Retrieve relevant document content to generate grounded responses.
- **Semantic search:** Use embeddings and vector similarity to retrieve relevant information.
- **Document similarity and plagiarism analysis:** Compare document content to identify potential overlap.
- **AI-generated text analysis:** Analyze text for characteristics associated with AI-generated content.
- **Local model execution:** Integrate locally running vision-language models to reduce reliance on external APIs.
- **Interactive interface:** Explore document intelligence features through Streamlit.

## Technology Stack

- Python
- Streamlit
- CLIP / OpenCLIP
- ChromaDB
- Qwen vision-language models
- PDF, image, and table processing libraries

## Project Structure

- `app.py` — Streamlit application
- `index_pdf.py` — PDF indexing entry point
- `src/embeddings/` — Multimodal embedding components
- `src/ingestion/` — PDF, image, and table processing
- `src/retrieval/` — Retrieval and vector-store components
- `src/generation/` — Local vision-language model integration
- `src/similarity/` — Similarity, plagiarism, and AI-text analysis
- `evaluation/` — Evaluation scripts and questions
- `tests/` — Automated tests
- `requirements.txt` — Python dependencies

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- pip
- A compatible local model setup for features requiring Qwen
- Sufficient memory and storage for the selected models

### Installation

Clone the repository:

```bash
git clone https://github.com/swarnimdeshmukh/AI-Powered-Multimodal-Document-Intelligence-System.git
cd AI-Powered-Multimodal-Document-Intelligence-System
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Some features may require additional local model downloads or configuration.

## Privacy and Model Execution

IntelliRAG is designed to support local model execution. Actual data handling depends on the configured components and any optional web-search functionality. Review model and component configurations before using sensitive documents.

## Evaluation

The `evaluation/` and `tests/` directories contain evaluation and testing resources. Run tests using:

```bash
python -m pytest tests/
```

## Limitations

- Plagiarism similarity is an indicator of textual overlap, not a definitive determination of plagiarism.
- AI-generated text detection is probabilistic and can produce false positives or false negatives.
- Results depend on document quality, model selection, retrieval configuration, and available computing resources.

## Author

**Swarnim Deshmukh**

GitHub: [@swarnimdeshmukh](https://github.com/swarnimdeshmukh)

---

*IntelliRAG — Multimodal Retrieval-Augmented Generation for Document Intelligence.*
