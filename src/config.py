from pathlib import Path

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
IMAGE_DIR = STORAGE_DIR / "extracted_images"
CHROMA_DIR = STORAGE_DIR / "chroma"

EMBEDDING_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
TOP_K = 5
