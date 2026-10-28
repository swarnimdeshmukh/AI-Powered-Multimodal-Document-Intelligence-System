import torch
import open_clip

from PIL import Image
from sentence_transformers import SentenceTransformer


class MultiModalEmbedder:

    def __init__(self):

        # Apple Silicon GPU for CLIP
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Loading CLIP on {self.device}...")

        self.clip_model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device=self.device,
            )
        )

        self.clip_model.eval()

        # Local text embedding model
        print("Loading text embedding model...")

        self.text_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Multimodal embedder ready.")

    # ---------------------------------------------------------
    # TEXT / TABLE EMBEDDINGS
    # ---------------------------------------------------------

    def embed_text(self, text):

        embedding = self.text_model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # ---------------------------------------------------------
    # IMAGE EMBEDDINGS
    # ---------------------------------------------------------

    def embed_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = self.preprocess(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():

            embedding = self.clip_model.encode_image(image)

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True,
        )

        return embedding[0].cpu().numpy().tolist()

    # ---------------------------------------------------------
    # CLIP TEXT EMBEDDING
    # Used when we need to search images using a question.
    # ---------------------------------------------------------

    def embed_clip_text(self, text):

        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        tokens = tokenizer([text]).to(self.device)

        with torch.no_grad():

            embedding = self.clip_model.encode_text(tokens)

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True,
        )

        return embedding[0].cpu().numpy().tolist()