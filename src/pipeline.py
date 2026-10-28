from src.embeddings.multimodal_embedder import MultiModalEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.query import MultimodalRetriever
from src.generation.local_vlm import LocalVLM


class MultimodalRAG:

    def __init__(self, top_k=5):

        print("Initializing Multimodal RAG...")

        self.top_k = top_k

        # ---------------------------------------------------------
        # MULTIMODAL EMBEDDER
        # ---------------------------------------------------------

        print("Loading multimodal embedder...")

        self.embedder = MultiModalEmbedder()

        # ---------------------------------------------------------
        # VECTOR STORE
        # ---------------------------------------------------------

        print("Loading vector store...")

        self.store = VectorStore()

        # ---------------------------------------------------------
        # RETRIEVER
        # ---------------------------------------------------------

        print("Initializing retriever...")

        self.retriever = MultimodalRetriever(
            self.embedder,
            self.store,
            top_k=self.top_k
        )

        # ---------------------------------------------------------
        # LOCAL VLM
        # ---------------------------------------------------------

        print("Loading local VLM...")

        self.generator = LocalVLM()

        print("Multimodal RAG initialized successfully.")

    # =============================================================
    # RETRIEVAL
    # =============================================================

    def retrieve(self, query):

        return self.retriever.search(
            query,
            top_k=self.top_k
        )

    # =============================================================
    # GENERATION
    # =============================================================

    def generate(self, query, results):

        if not results:

            return (
                "I could not find relevant information "
                "in the indexed documents."
            )

        # Use the EXISTING retriever methods.
        context = self.retriever.get_context(
            results
        )

        image_paths = self.retriever.get_image_paths(
            results
        )

        # IMPORTANT:
        # LocalVLM.generate() requires:
        #
        # question
        # context_text
        # image_paths

        answer = self.generator.generate(
            query,
            context,
            image_paths
        )

        return answer

    # =============================================================
    # ASK
    # =============================================================

    def ask(self, query):

        print("\n" + "=" * 60)
        print("QUERY")
        print("=" * 60)

        print(query)

        # ---------------------------------------------------------
        # RETRIEVAL
        # ---------------------------------------------------------

        print("\nRetrieving relevant information...")

        results = self.retrieve(
            query
        )

        print("Retrieval completed.")

        # ---------------------------------------------------------
        # DISPLAY RETRIEVAL INFORMATION
        # ---------------------------------------------------------

        if isinstance(results, dict):

            text_items = results.get(
                "text",
                []
            )

            image_items = results.get(
                "images",
                []
            )

            print(
                f"Retrieved text/table items : "
                f"{len(text_items)}"
            )

            print(
                f"Retrieved images           : "
                f"{len(image_items)}"
            )

        # ---------------------------------------------------------
        # GENERATION
        # ---------------------------------------------------------

        print("\nGenerating answer...")

        answer = self.generate(
            query,
            results
        )

        # ---------------------------------------------------------
        # SOURCES
        # ---------------------------------------------------------

        sources = self.retriever.get_sources(
            results
        )

        # ---------------------------------------------------------
        # IMAGE PATHS
        # ---------------------------------------------------------

        image_paths = self.retriever.get_image_paths(
            results
        )

        return {
            "query": query,
            "answer": answer,
            "results": results,
            "sources": sources,
            "images": image_paths
        }


# ================================================================
# MAIN
# ================================================================

def main():

    rag = MultimodalRAG(
        top_k=5
    )

    print("\n" + "=" * 60)
    print("MULTIMODAL RAG READY")
    print("=" * 60)

    print("Type your question.")
    print("Type 'exit' to quit.\n")

    while True:

        try:

            query = input("You: ").strip()

        except KeyboardInterrupt:

            print("\nExiting...")
            break

        if not query:

            continue

        if query.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print("Exiting...")
            break

        try:

            result = rag.ask(
                query
            )

            # -----------------------------------------------------
            # ANSWER
            # -----------------------------------------------------

            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)

            print(
                result["answer"]
            )

            # -----------------------------------------------------
            # SOURCES
            # -----------------------------------------------------

            if result["sources"]:

                print("\n" + "-" * 60)
                print("SOURCES")
                print("-" * 60)

                for source in result["sources"]:

                    if isinstance(source, dict):

                        modality = source.get(
                            "modality",
                            "unknown"
                        )

                        page = source.get(
                            "page",
                            "?"
                        )

                        print(
                            f"- {modality} | Page {page}"
                        )

                    else:

                        print(
                            f"- {source}"
                        )

            # -----------------------------------------------------
            # IMAGES
            # -----------------------------------------------------

            if result["images"]:

                print("\n" + "-" * 60)
                print("RELATED IMAGES")
                print("-" * 60)

                for image in result["images"]:

                    print(
                        f"- {image}"
                    )

            print()

        except Exception as e:

            print("\n" + "=" * 60)
            print("ERROR")
            print("=" * 60)

            print(
                type(e).__name__
            )

            print(
                str(e)
            )

            print()


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()