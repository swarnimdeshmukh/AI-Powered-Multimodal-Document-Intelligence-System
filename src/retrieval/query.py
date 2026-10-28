import re


class MultimodalRetriever:

    def __init__(self, embedder, store, top_k=5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    # =========================================================
    # Detect explicit figure references
    # =========================================================
    def get_figure_number(self, question):
        match = re.search(
            r"(?:figure|fig\.)\s*(\d+)",
            question.lower()
        )

        if match:
            return match.group(1)

        return None

    # =========================================================
    # Detect important concepts in the question
    # =========================================================
    def detect_concept(self, question):
        q = question.lower()

        # RCR + RCG
        if "rcr" in q and "rcg" in q:
            return "rcr_rcg"

        # Full names
        if "representation compression via retrieval" in q:
            return "rcr"

        if "representation compression via generation" in q:
            return "rcg"

        # VDocRetriever method-specific questions
        if "vdocretriever" in q:
            if (
                "retrieval method" in q
                or "retrieval technique" in q
                or "how does" in q
                or "how does it" in q
                or "retrieve" in q
                or "similarity" in q
                or "search" in q
            ):
                return "retrieval_method"

            return "vdocretriever"

        # VDocGenerator
        if "vdocgenerator" in q:
            return "vdocgenerator"

        # Generic retrieval-method question
        if (
            "retrieval method" in q
            or "retrieval technique" in q
            or "maximum inner product" in q
            or "how does it retrieve" in q
            or "what search" in q
        ):
            return "retrieval_method"

        # Dynamic resolution
        if (
            "dynamic high-resolution" in q
            or "dynamic high resolution" in q
        ):
            return "dynamic_resolution"

        return None

    # =========================================================
    # Find pages containing a specific figure
    # =========================================================
    def find_figure_pages(self, figure_number):

        if not figure_number:
            return set()

        patterns = [
            f"figure {figure_number}",
            f"figure{figure_number}",
            f"fig. {figure_number}",
        ]

        pages = set()

        try:
            data = self.store.text_collection.get(
                include=["documents", "metadatas"]
            )

            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            for document, metadata in zip(
                documents,
                metadatas
            ):
                text = (document or "").lower()

                if any(
                    pattern in text
                    for pattern in patterns
                ):
                    page = metadata.get("page")

                    if page is not None:
                        pages.add(str(page))

        except Exception as e:
            print(
                f"Warning: figure lookup failed: {e}"
            )

        return pages

    # =========================================================
    # Find pages containing a specific concept
    # =========================================================
    def find_concept_pages(self, concept):

        if not concept:
            return set()

        concept_patterns = {

            "rcr_rcg": [
                "representation compression via retrieval",
                "representation compression via generation",
                "(rcr)",
                "(rcg)",
            ],

            "rcr": [
                "representation compression via retrieval",
                "(rcr)",
            ],

            "rcg": [
                "representation compression via generation",
                "(rcg)",
            ],

            "vdocretriever": [
                "vdocretriever",
            ],

            "vdocgenerator": [
                "vdocgenerator",
            ],

            "retrieval_method": [
                "maximum inner product search",
                "maximum inner product",
                "highest similarity",
                "similarity scores",
            ],

            "dynamic_resolution": [
                "dynamic high resolution",
                "dynamic high-resolution",
            ],
        }

        patterns = concept_patterns.get(
            concept,
            []
        )

        if not patterns:
            return set()

        pages = set()

        try:
            data = self.store.text_collection.get(
                include=["documents", "metadatas"]
            )

            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            for document, metadata in zip(
                documents,
                metadatas
            ):
                text = (document or "").lower()

                if any(
                    pattern in text
                    for pattern in patterns
                ):
                    page = metadata.get("page")

                    if page is not None:
                        pages.add(str(page))

        except Exception as e:
            print(
                f"Warning: concept lookup failed: {e}"
            )

        return pages

    # =========================================================
    # Main retrieval function
    # =========================================================
    def search(self, question, top_k=None):

        if top_k is None:
            top_k = self.top_k

        print("\n" + "=" * 60)
        print("RETRIEVING CONTEXT")
        print("=" * 60)

        # -----------------------------------------------------
        # Detect figure and concept
        # -----------------------------------------------------

        figure_number = self.get_figure_number(
            question
        )

        concept = self.detect_concept(
            question
        )

        if figure_number:
            print(
                f"Detected Figure {figure_number}"
            )

        if concept:
            print(
                f"Detected concept: {concept}"
            )

        # -----------------------------------------------------
        # Semantic text retrieval
        # -----------------------------------------------------

        text_embedding = self.embedder.embed_text(
            question
        )

        candidate_k = max(
            top_k * 5,
            20
        )

        text_results = self.store.search_text(
            embedding=text_embedding,
            top_k=candidate_k,
        )

        documents = text_results.get(
            "documents",
            [[]]
        )[0]

        metadatas = text_results.get(
            "metadatas",
            [[]]
        )[0]

        distances = text_results.get(
            "distances",
            [[]]
        )[0]

        text_items = []

        for i, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[i]
                if i < len(metadatas)
                else {}
            )

            distance = (
                distances[i]
                if i < len(distances)
                else None
            )

            if not document:
                continue

            # Ignore tiny/noisy chunks
            if len(document.strip()) < 30:
                continue

            text_items.append({
                "document": document,
                "metadata": metadata,
                "distance": distance,
            })

        # -----------------------------------------------------
        # Determine relevant pages
        # -----------------------------------------------------

        target_pages = set()

        # Exact figure routing
        if figure_number:

            figure_pages = (
                self.find_figure_pages(
                    figure_number
                )
            )

            print(
                f"Figure pages found: "
                f"{sorted(figure_pages)}"
            )

            target_pages.update(
                figure_pages
            )

        # Concept routing
        if concept:

            concept_pages = (
                self.find_concept_pages(
                    concept
                )
            )

            print(
                f"Concept pages found: "
                f"{sorted(concept_pages)}"
            )

            target_pages.update(
                concept_pages
            )

        # -----------------------------------------------------
        # Prioritize target pages
        # -----------------------------------------------------

        if target_pages:

            targeted_items = []
            other_items = []

            for item in text_items:

                page = str(
                    item["metadata"].get(
                        "page",
                        ""
                    )
                )

                if page in target_pages:
                    targeted_items.append(item)
                else:
                    other_items.append(item)

            text_items = (
                targeted_items +
                other_items
            )

        # -----------------------------------------------------
        # For highly specific concepts, remove
        # unrelated pages completely
        # -----------------------------------------------------

        if (
            concept in {
                "rcr_rcg",
                "rcr",
                "rcg",
                "retrieval_method",
            }
            and target_pages
        ):

            focused_items = []

            for item in text_items:

                page = str(
                    item["metadata"].get(
                        "page",
                        ""
                    )
                )

                if page in target_pages:
                    focused_items.append(item)

            if focused_items:
                text_items = focused_items

        # -----------------------------------------------------
        # Remove duplicate chunks
        # -----------------------------------------------------

        seen = set()
        clean_items = []

        for item in text_items:

            key = (
                str(
                    item["metadata"].get(
                        "page",
                        ""
                    )
                ),
                item["document"].strip()[:200],
            )

            if key in seen:
                continue

            seen.add(key)
            clean_items.append(item)

        text_items = clean_items[:top_k]

        # =====================================================
        # IMAGE RETRIEVAL
        # =====================================================

        clip_embedding = (
            self.embedder.embed_clip_text(
                question
            )
        )

        image_results = self.store.search_images(
            embedding=clip_embedding,
            top_k=top_k,
        )

        image_documents = image_results.get(
            "documents",
            [[]]
        )[0]

        image_metadatas = image_results.get(
            "metadatas",
            [[]]
        )[0]

        image_distances = image_results.get(
            "distances",
            [[]]
        )[0]

        image_items = []

        for i, document in enumerate(
            image_documents
        ):

            metadata = (
                image_metadatas[i]
                if i < len(image_metadatas)
                else {}
            )

            distance = (
                image_distances[i]
                if i < len(image_distances)
                else None
            )

            image_items.append({
                "document": document,
                "metadata": metadata,
                "distance": distance,
            })

        # -----------------------------------------------------
        # Prioritize relevant page images
        # -----------------------------------------------------

        if target_pages:

            matching_images = []
            other_images = []

            for item in image_items:

                page = str(
                    item["metadata"].get(
                        "page",
                        ""
                    )
                )

                if page in target_pages:
                    matching_images.append(item)
                else:
                    other_images.append(item)

            image_items = (
                matching_images +
                other_images
            )

        print(
            f"Retrieved text/table items : "
            f"{len(text_items)}"
        )

        print(
            f"Retrieved images           : "
            f"{len(image_items)}"
        )

        return {
            "text": text_items,
            "images": image_items,
        }

    # =========================================================
    # Build context for local VLM
    # =========================================================
    def get_context(self, results):

        context_parts = []

        for item in results.get(
            "text",
            []
        ):

            metadata = item["metadata"]

            page = metadata.get(
                "page",
                "?"
            )

            modality = metadata.get(
                "modality",
                metadata.get(
                    "type",
                    "text"
                )
            )

            context_parts.append(
                f"[{modality} | Page {page}]\n"
                f"{item['document']}"
            )

        return "\n\n".join(
            context_parts
        )

    # =========================================================
    # Get source information
    # =========================================================
    def get_sources(self, results):

        sources = []

        for item in results.get(
            "text",
            []
        ):

            metadata = item["metadata"]

            page = metadata.get(
                "page",
                "?"
            )

            modality = metadata.get(
                "modality",
                metadata.get(
                    "type",
                    "text"
                )
            )

            sources.append({
                "modality": modality,
                "page": page,
            })

        return sources

    # =========================================================
    # Get image paths
    # =========================================================
    def get_image_paths(self, results):

        paths = []

        for item in results.get(
            "images",
            []
        ):

            metadata = item["metadata"]

            path = (
                metadata.get("source")
                or metadata.get("path")
            )

            if path:
                paths.append(path)

        return paths