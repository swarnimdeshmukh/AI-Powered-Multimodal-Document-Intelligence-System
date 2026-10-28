import chromadb
from pathlib import Path


class VectorStore:

    def __init__(self, persist_dir="storage/chroma"):

        Path(persist_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        self.persist_dir = persist_dir

        self.client = chromadb.PersistentClient(
            path=persist_dir
        )

        self.text_collection = (
            self.client.get_or_create_collection(
                name="text_documents"
            )
        )

        self.image_collection = (
            self.client.get_or_create_collection(
                name="image_documents"
            )
        )

    # =========================================================
    # ADD TEXT
    # =========================================================

    def add_text(
        self,
        embedding,
        document,
        metadata,
        item_id
    ):

        self.text_collection.add(
            ids=[item_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    # =========================================================
    # ADD IMAGE
    # =========================================================

    def add_image(
        self,
        embedding,
        document,
        metadata,
        item_id
    ):

        self.image_collection.add(
            ids=[item_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    # =========================================================
    # SEARCH TEXT
    # =========================================================

    def search_text(
        self,
        embedding,
        top_k=5
    ):

        count = self.text_collection.count()

        if count == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        top_k = min(
            top_k,
            count
        )

        return self.text_collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

    # =========================================================
    # SEARCH IMAGES
    # =========================================================

    def search_images(
        self,
        embedding,
        top_k=5
    ):

        count = self.image_collection.count()

        if count == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        top_k = min(
            top_k,
            count
        )

        return self.image_collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

    # =========================================================
    # CLEAR CURRENT INDEX
    # =========================================================

    def clear(self):

        print("\nClearing existing vector index...")

        # Delete existing collections
        try:
            self.client.delete_collection(
                name="text_documents"
            )
        except Exception:
            pass

        try:
            self.client.delete_collection(
                name="image_documents"
            )
        except Exception:
            pass

        # Recreate collections
        self.text_collection = (
            self.client.get_or_create_collection(
                name="text_documents"
            )
        )

        self.image_collection = (
            self.client.get_or_create_collection(
                name="image_documents"
            )
        )

        print("Existing vector index cleared.")

    # =========================================================
    # COUNTS
    # =========================================================

    def text_count(self):

        return self.text_collection.count()

    def image_count(self):

        return self.image_collection.count()