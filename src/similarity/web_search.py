from ddgs import DDGS
import re
import time


class WebPlagiarismSearch:
    """
    Web source discovery for plagiarism checking.

    Search strategy:
    1. Try an exact quoted search.
    2. If no results are found, try a broader unquoted search.
    3. If that also fails, try a shorter distinctive query.
    4. Keep search snippets separately because they can contain
       the exact matching text.
    5. Attempt to extract the full web page.
    6. No API key required.
    """

    def __init__(
        self,
        max_results_per_query=5,
        max_sources=20,
        timeout=15,
    ):
        self.max_results_per_query = max_results_per_query
        self.max_sources = max_sources
        self.timeout = timeout

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    @staticmethod
    def clean_text(text):
        if not text:
            return ""

        text = text.replace("\x00", " ")

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ========================================================
    # CREATE SEARCH QUERY
    # ========================================================

    def create_query(self, passage):
        text = self.clean_text(passage)

        words = text.split()

        if len(words) > 16:
            words = words[:16]

        if len(words) < 5:
            return ""

        query_text = " ".join(words)

        return f'"{query_text}"'

    # ========================================================
    # CREATE FALLBACK QUERIES
    # ========================================================

    def create_fallback_queries(self, passage):
        """
        Create broader queries when the exact quoted query
        produces no search results.
        """

        text = self.clean_text(passage)

        if not text:
            return []

        words = text.split()

        queries = []

        # ----------------------------------------------------
        # Fallback 1:
        # First 10 words without quotation marks.
        # ----------------------------------------------------
        if len(words) >= 10:
            queries.append(
                " ".join(words[:10])
            )

        # ----------------------------------------------------
        # Fallback 2:
        # Middle/central distinctive portion.
        # This avoids relying only on the beginning of a
        # sentence.
        # ----------------------------------------------------
        if len(words) >= 14:
            start = max(
                0,
                min(
                    len(words) - 8,
                    len(words) // 2 - 4
                )
            )

            middle_query = " ".join(
                words[start:start + 8]
            )

            if middle_query not in queries:
                queries.append(middle_query)

        # ----------------------------------------------------
        # Fallback 3:
        # Last 8 words.
        # ----------------------------------------------------
        if len(words) >= 8:
            last_query = " ".join(
                words[-8:]
            )

            if last_query not in queries:
                queries.append(last_query)

        # ----------------------------------------------------
        # Remove extremely generic queries.
        # ----------------------------------------------------
        cleaned_queries = []

        for query in queries:
            query = self.clean_text(query)

            if len(query.split()) >= 5:
                cleaned_queries.append(query)

        return cleaned_queries

    # ========================================================
    # SEARCH WEB
    # ========================================================

    def search_web(self, query):
        if not query:
            return []

        try:
            with DDGS(
                timeout=self.timeout
            ) as searcher:

                results = searcher.text(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    max_results=self.max_results_per_query
                )

                return list(results)

        except Exception as e:
            print(
                f"[Web Search Error] {e}"
            )

            return []

    # ========================================================
    # SEARCH WITH FALLBACKS
    # ========================================================

    def search_web_with_fallback(self, passage):
        """
        Search using the original exact-query strategy first.

        If the exact query returns no results, progressively
        broaden the search.

        Returns:
            (results, query_used)
        """

        exact_query = self.create_query(
            passage
        )

        if not exact_query:
            return [], ""

        # ----------------------------------------------------
        # PASS 1: Exact quoted search
        # ----------------------------------------------------

        results = self.search_web(
            exact_query
        )

        if results:
            return results, exact_query

        # ----------------------------------------------------
        # PASS 2+: Broader fallback searches
        # ----------------------------------------------------

        fallback_queries = self.create_fallback_queries(
            passage
        )

        for fallback_query in fallback_queries:

            print(
                f"[Web Search Fallback] {fallback_query}"
            )

            results = self.search_web(
                fallback_query
            )

            if results:
                return results, fallback_query

            time.sleep(0.10)

        return [], exact_query

    # ========================================================
    # EXTRACT WEB PAGE
    # ========================================================

    def extract_page(self, url):
        if not url:
            return ""

        try:
            with DDGS(
                timeout=self.timeout
            ) as searcher:

                result = searcher.extract(
                    url,
                    fmt="text_plain"
                )

            if not result:
                return ""

            if isinstance(
                result,
                dict
            ):

                content = result.get(
                    "content",
                    ""
                )

            else:

                content = str(
                    result
                )

            return self.clean_text(
                content
            )

        except Exception as e:
            print(
                f"[Page Extraction Error] "
                f"{url} -> {e}"
            )

            return ""

    # ========================================================
    # SEARCH ONE PASSAGE
    # ========================================================

    def search_passage(self, passage):
        results, query_used = (
            self.search_web_with_fallback(
                passage
            )
        )

        if not results:
            return []

        sources = []

        for result in results:

            if not isinstance(
                result,
                dict
            ):
                continue

            url = result.get(
                "href"
            )

            if not url:
                url = result.get(
                    "url"
                )

            if not url:
                continue

            title = self.clean_text(
                result.get(
                    "title",
                    "Online Source"
                )
            )

            snippet = self.clean_text(
                result.get(
                    "body",
                    ""
                )
            )

            sources.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "query": query_used
                }
            )

        return sources

    # ========================================================
    # SEARCH DOCUMENT
    # ========================================================

    def search_document(
        self,
        passages,
        progress_callback=None
    ):

        if not passages:
            return []

        # ----------------------------------------------------
        # Search more passages for better coverage.
        # ----------------------------------------------------

        max_queries = min(
            40,
            len(passages)
        )

        if len(passages) > max_queries:

            step = max(
                1,
                len(passages) // max_queries
            )

            selected_passages = (
                passages[::step]
            )

            selected_passages = (
                selected_passages[
                    :max_queries
                ]
            )

        else:

            selected_passages = passages

        source_documents = []

        visited_urls = set()

        total = len(
            selected_passages
        )

        for index, passage in enumerate(
            selected_passages
        ):

            if progress_callback:

                progress_callback(
                    index + 1,
                    total
                )

            try:

                results = (
                    self.search_passage(
                        passage["text"]
                    )
                )

            except Exception as e:

                print(
                    f"[Passage Search Error] "
                    f"{e}"
                )

                continue

            for result in results:

                url = result.get(
                    "url"
                )

                if not url:
                    continue

                title = result.get(
                    "title",
                    "Online Source"
                )

                snippet = result.get(
                    "snippet",
                    ""
                )

                # =================================================
                # KEEP SEARCH SNIPPET AS ITS OWN SOURCE
                # =================================================

                if snippet:

                    source_documents.append(
                        {
                            "text": snippet,
                            "source": title,
                            "url": url,
                            "page": None,
                            "snippet": snippet,
                            "source_type": "search_snippet"
                        }
                    )

                # -------------------------------------------------
                # Avoid repeatedly downloading same URL.
                # -------------------------------------------------

                if url in visited_urls:
                    continue

                visited_urls.add(
                    url
                )

                # -------------------------------------------------
                # Extract full webpage.
                # -------------------------------------------------

                page_text = (
                    self.extract_page(
                        url
                    )
                )

                if page_text:

                    source_documents.append(
                        {
                            "text": page_text,
                            "source": title,
                            "url": url,
                            "page": None,
                            "snippet": snippet,
                            "source_type": "webpage"
                        }
                    )

                if (
                    len(source_documents)
                    >= self.max_sources * 2
                ):
                    break

            if (
                len(source_documents)
                >= self.max_sources * 2
            ):
                break

            time.sleep(
                0.20
            )

        # ----------------------------------------------------
        # Deduplicate source entries.
        # ----------------------------------------------------

        unique = []

        seen = set()

        for source in source_documents:

            key = (
                source.get(
                    "url",
                    ""
                ),
                self.clean_text(
                    source.get(
                        "text",
                        ""
                    )
                )[:500]
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique.append(
                source
            )

        # ----------------------------------------------------
        # Keep strongest/most useful source entries.
        # ----------------------------------------------------

        return unique[
            :self.max_sources * 2
        ]

    # ========================================================
    # PREPARE SOURCES FOR TEXT MATCHING
    # ========================================================

    def prepare_for_ai(
        self,
        source_documents,
        checker
    ):

        all_passages = []

        for source in source_documents:

            text = source.get(
                "text",
                ""
            )

            if not text:
                continue

            # ------------------------------------------------
            # Search snippets should NOT be grouped with
            # unrelated sentences.
            # ------------------------------------------------

            if source.get(
                "source_type"
            ) == "search_snippet":

                normalized = (
                    checker.normalize_text(
                        text
                    )
                )

                if len(
                    normalized.split()
                ) >= 5:

                    all_passages.append(
                        {
                            "text": text,
                            "normalized": normalized,
                            "page": None,
                            "source": source.get(
                                "source",
                                "Online Source"
                            ),
                            "url": source.get(
                                "url"
                            ),
                            "source_type": (
                                "search_snippet"
                            )
                        }
                    )

                continue

            # ------------------------------------------------
            # Full webpage.
            # ------------------------------------------------

            passages = (
                checker.prepare_source_text(
                    source_text=text,
                    source_name=source.get(
                        "source",
                        "Online Source"
                    ),
                    source_url=source.get(
                        "url"
                    ),
                    source_page=source.get(
                        "page"
                    )
                )
            )

            for passage in passages:

                passage[
                    "source_type"
                ] = "webpage"

            all_passages.extend(
                passages
            )

        return all_passages