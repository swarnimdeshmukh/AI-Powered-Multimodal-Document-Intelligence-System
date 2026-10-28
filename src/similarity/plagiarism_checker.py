from pathlib import Path
import re
from difflib import SequenceMatcher

import fitz


class PlagiarismChecker:
    """
    Local text similarity / plagiarism detector.

    Plagiarism detection is based on:
        - exact word sequences
        - n-gram overlap
        - fuzzy textual matching
        - longest common word sequence

    Semantic similarity alone is NEVER considered plagiarism.
    AI detection is handled separately by ai_detector.py.
    """

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.20,
        high_similarity_threshold=0.55,
        threshold=None,
        high_threshold=None,
    ):
        if threshold is not None:
            similarity_threshold = threshold

        if high_threshold is not None:
            high_similarity_threshold = high_threshold

        self.model_name = model_name

        self.similarity_threshold = float(
            similarity_threshold
        )

        self.high_similarity_threshold = float(
            high_similarity_threshold
        )

        # Minimum continuous words required for a real
        # textual similarity match.
        self.minimum_match_words = 5

        # 12+ consecutive words = very strong match.
        self.high_match_words = 12

        self.ngram_size = 4

        self.generic_phrases = {
            "in this paper",
            "this paper presents",
            "this study presents",
            "the proposed method",
            "the proposed model",
            "the proposed system",
            "the results show",
            "results show that",
            "based on the results",
            "in recent years",
            "according to the results",
            "as shown in figure",
            "as shown in table",
            "the following results",
            "machine learning algorithms",
            "machine learning models",
            "deep learning models",
            "artificial intelligence techniques",
        }

    # ============================================================
    # NORMALIZE TEXT
    # ============================================================

    @staticmethod
    def normalize_text(text):
        if not text:
            return ""

        text = str(text).lower()

        # Fix words broken by PDF line wrapping.
        text = re.sub(
            r"-\s*\n\s*",
            "",
            text
        )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        # Remove punctuation.
        text = re.sub(
            r"[^\w\s]",
            " ",
            text
        )

        # Normalize whitespace again.
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ============================================================
    # TOKENIZE
    # ============================================================

    def tokenize(self, text):
        normalized = self.normalize_text(
            text
        )

        if not normalized:
            return []

        return normalized.split()

    # ============================================================
    # REMOVE GENERIC PHRASES
    # ============================================================

    def remove_generic_phrases(self, text):
        normalized = self.normalize_text(
            text
        )

        if not normalized:
            return ""

        for phrase in self.generic_phrases:
            normalized = normalized.replace(
                phrase,
                " "
            )

        return re.sub(
            r"\s+",
            " ",
            normalized
        ).strip()

    # ============================================================
    # EXTRACT PDF SENTENCES
    # ============================================================

    def extract_pdf_sentences(
        self,
        pdf_path
    ):
        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        doc = fitz.open(
            str(pdf_path)
        )

        sentences = []

        for page_number, page in enumerate(
            doc,
            start=1
        ):
            text = page.get_text(
                "text"
            )

            if not text:
                continue

            # Join PDF line breaks.
            text = re.sub(
                r"-\s*\n\s*",
                "",
                text
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            if not text:
                continue

            # Split into sentences.
            raw_sentences = re.split(
                r"(?<=[.!?])\s+",
                text
            )

            for sentence in raw_sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                normalized = self.normalize_text(
                    sentence
                )

                # Ignore tiny fragments.
                if len(
                    normalized.split()
                ) < 5:
                    continue

                sentences.append(
                    {
                        "text": sentence,
                        "normalized": normalized,
                        "page": page_number,
                    }
                )

        doc.close()

        return sentences

    # ============================================================
    # CREATE DOCUMENT PASSAGES
    # ============================================================

    def create_passages(
        self,
        sentences,
        sentences_per_passage=3
    ):
        passages = []

        for i in range(
            0,
            len(sentences),
            sentences_per_passage
        ):
            group = sentences[
                i:i + sentences_per_passage
            ]

            if not group:
                continue

            text = " ".join(
                item["text"]
                for item in group
            )

            passages.append(
                {
                    "text": text,

                    "normalized": self.normalize_text(
                        text
                    ),

                    "page": group[0]["page"],

                    "sentence_start": i,

                    "sentence_end": (
                        i
                        + len(group)
                        - 1
                    ),
                }
            )

        return passages

    # ============================================================
    # PREPARE SOURCE TEXT
    # ============================================================

    def prepare_source_text(
        self,
        source_text,
        source_name="Online Source",
        source_url=None,
        source_page=None
    ):
        if not source_text:
            return []

        source_text = re.sub(
            r"\s+",
            " ",
            str(source_text)
        ).strip()

        if not source_text:
            return []

        raw_sentences = re.split(
            r"(?<=[.!?])\s+",
            source_text
        )

        sentences = []

        for sentence in raw_sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            normalized = self.normalize_text(
                sentence
            )

            if len(
                normalized.split()
            ) < 5:
                continue

            sentences.append(
                {
                    "text": sentence,

                    "normalized": normalized,

                    "page": source_page,

                    "source": source_name,

                    "url": source_url,
                }
            )

        passages = []

        for i in range(
            0,
            len(sentences),
            3
        ):
            group = sentences[
                i:i + 3
            ]

            if not group:
                continue

            text = " ".join(
                item["text"]
                for item in group
            )

            passages.append(
                {
                    "text": text,

                    "normalized": self.normalize_text(
                        text
                    ),

                    "page": source_page,

                    "source": source_name,

                    "url": source_url,
                }
            )

        return passages

    # ============================================================
    # LONGEST COMMON WORD SEQUENCE
    # ============================================================

    def longest_common_sequence(
        self,
        text1,
        text2
    ):
        tokens1 = self.tokenize(
            text1
        )

        tokens2 = self.tokenize(
            text2
        )

        if not tokens1 or not tokens2:
            return {
                "length": 0,
                "text": "",
                "document_start": 0,
                "source_start": 0,
            }

        matcher = SequenceMatcher(
            None,
            tokens1,
            tokens2,
            autojunk=False
        )

        match = matcher.find_longest_match(
            0,
            len(tokens1),
            0,
            len(tokens2)
        )

        length = match.size

        if length == 0:
            return {
                "length": 0,
                "text": "",
                "document_start": 0,
                "source_start": 0,
            }

        return {
            "length": length,

            "text": " ".join(
                tokens1[
                    match.a:
                    match.a + length
                ]
            ),

            "document_start": match.a,

            "source_start": match.b,
        }

    # ============================================================
    # N-GRAM OVERLAP
    # ============================================================

    def ngram_overlap(
        self,
        text1,
        text2
    ):
        tokens1 = self.tokenize(
            text1
        )

        tokens2 = self.tokenize(
            text2
        )

        n = self.ngram_size

        if (
            len(tokens1) < n
            or len(tokens2) < n
        ):
            return 0.0

        grams1 = {
            tuple(
                tokens1[
                    i:i + n
                ]
            )
            for i in range(
                len(tokens1) - n + 1
            )
        }

        grams2 = {
            tuple(
                tokens2[
                    i:i + n
                ]
            )
            for i in range(
                len(tokens2) - n + 1
            )
        }

        if not grams1:
            return 0.0

        return (
            len(
                grams1.intersection(
                    grams2
                )
            )
            /
            len(grams1)
        )

    # ============================================================
    # FUZZY SIMILARITY
    # ============================================================

    def fuzzy_similarity(
        self,
        text1,
        text2
    ):
        text1 = self.remove_generic_phrases(
            text1
        )

        text2 = self.remove_generic_phrases(
            text2
        )

        if not text1 or not text2:
            return 0.0

        return SequenceMatcher(
            None,
            text1,
            text2,
            autojunk=False
        ).ratio()

    # ============================================================
    # TEXTUAL MATCH SCORE
    # ============================================================

    def textual_match_score(
        self,
        document_text,
        source_text
    ):
        document_tokens = self.tokenize(
            document_text
        )

        source_tokens = self.tokenize(
            source_text
        )

        if (
            not document_tokens
            or not source_tokens
        ):
            return {
                "score": 0.0,
                "longest_match": 0,
                "ngram_score": 0.0,
                "fuzzy_score": 0.0,
                "matched_text": "",
            }

        longest = self.longest_common_sequence(
            document_text,
            source_text
        )

        longest_words = longest[
            "length"
        ]

        ngram_score = self.ngram_overlap(
            document_text,
            source_text
        )

        fuzzy_score = self.fuzzy_similarity(
            document_text,
            source_text
        )

        document_word_count = max(
            len(document_tokens),
            1
        )

        # Percentage of the document passage that
        # belongs to the longest continuous match.
        sequence_coverage = (
            longest_words
            /
            document_word_count
        )

        # Main similarity score.
        #
        # Continuous exact text receives the strongest weight.
        score = (
            sequence_coverage * 0.70
            +
            ngram_score * 0.20
            +
            fuzzy_score * 0.10
        )

        # Fuzzy similarity alone can never trigger plagiarism.
        if (
            longest_words
            < self.minimum_match_words
        ):
            score = 0.0

        return {
            "score": float(
                min(
                    score,
                    1.0
                )
            ),

            "longest_match": (
                longest_words
            ),

            "ngram_score": float(
                ngram_score
            ),

            "fuzzy_score": float(
                fuzzy_score
            ),

            "matched_text": (
                longest["text"]
            ),
        }

    # ============================================================
    # COMPARE DOCUMENT WITH SOURCES
    # ============================================================

    def compare_with_sources(
        self,
        document_passages,
        source_passages
    ):
        """
        Compare document passages against online sources.

        IMPORTANT:
        No 4-gram pre-filter is used.

        The previous optimization could discard valid comparisons
        before textual_match_score() was called.

        The actual textual matcher requires a minimum continuous
        sequence of words, so it is used directly.
        """

        if not document_passages:
            return []

        if not source_passages:
            return []

        matches = []

        for document_passage in document_passages:

            document_text = document_passage.get(
                "text",
                ""
            )

            if not document_text:
                continue

            best = None

            for source_passage in source_passages:

                source_text = source_passage.get(
                    "text",
                    ""
                )

                if not source_text:
                    continue

                result = self.textual_match_score(
                    document_text,
                    source_text
                )

                longest_words = result.get(
                    "longest_match",
                    0
                )

                score = result.get(
                    "score",
                    0.0
                )

                # Minimum continuous match requirement.
                if (
                    longest_words
                    < self.minimum_match_words
                ):
                    continue

                candidate = {
                    **result,
                    "source_passage": source_passage,
                }

                # Select the source with the longest
                # continuous overlap.
                if best is None:

                    best = candidate

                elif (
                    longest_words
                    >
                    best["longest_match"]
                ):

                    best = candidate

                # If LCS is equal, choose higher score.
                elif (
                    longest_words
                    ==
                    best["longest_match"]
                    and
                    score
                    >
                    best["score"]
                ):

                    best = candidate

            if best is None:
                continue

            longest_words = best[
                "longest_match"
            ]

            score = best[
                "score"
            ]

            ngram_score = best.get(
                "ngram_score",
                0.0
            )

            # ====================================================
            # MATCH CLASSIFICATION
            # ====================================================

            # 10+ consecutive words:
            # strong textual evidence.
            strong_match = (
                longest_words >= 10
            )

            # 7-9 consecutive words:
            # meaningful overlap.
            moderate_match = (
                longest_words >= 7
            )

            # 5-6 consecutive words:
            # require supporting n-gram overlap.
            shorter_match = (
                longest_words >= 5
                and
                ngram_score >= 0.25
            )

            if not (
                strong_match
                or moderate_match
                or shorter_match
            ):
                continue

            if strong_match:
                level = "high"
            else:
                level = "moderate"

            source = best[
                "source_passage"
            ]

            matches.append(
                {
                    "text": document_passage[
                        "text"
                    ],

                    "page": document_passage[
                        "page"
                    ],

                    "similarity": float(
                        score
                    ),

                    "semantic_similarity": 0.0,

                    "lexical_similarity": float(
                        ngram_score
                    ),

                    "ngram_similarity": float(
                        ngram_score
                    ),

                    "fuzzy_similarity": float(
                        best.get(
                            "fuzzy_score",
                            0.0
                        )
                    ),

                    "longest_match_words": (
                        longest_words
                    ),

                    "matched_text": (
                        best.get(
                            "matched_text",
                            ""
                        )
                    ),

                    "matched_words": (
                        longest_words
                    ),

                    "level": level,

                    "source": source.get(
                        "source",
                        "Online Source"
                    ),

                    "url": source.get(
                        "url"
                    ),

                    "reference_page": source.get(
                        "page"
                    ),

                    "reference_text": source.get(
                        "text",
                        ""
                    ),
                }
            )

        return matches

    # ============================================================
    # REMOVE DUPLICATE MATCHES
    # ============================================================

    def remove_duplicate_matches(
        self,
        matches
    ):
        if not matches:
            return []

        matches = sorted(
            matches,

            key=lambda x: (
                x.get(
                    "longest_match_words",
                    0
                ),

                x.get(
                    "similarity",
                    0
                )
            ),

            reverse=True
        )

        selected = []

        seen = set()

        for match in matches:

            page = match.get(
                "page"
            )

            matched = self.normalize_text(
                match.get(
                    "matched_text",
                    ""
                )
            )

            key = (
                page,
                matched
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            selected.append(
                match
            )

        return selected

    # ============================================================
    # ANALYZE DOCUMENT
    # ============================================================

    def analyze_document(
        self,
        pdf_path,
        source_passages=None
    ):
        sentences = self.extract_pdf_sentences(
            pdf_path
        )

        passages = self.create_passages(
            sentences
        )

        if not source_passages:
            return {
                "total_sentences": len(
                    sentences
                ),

                "total_passages": len(
                    passages
                ),

                "matches": [],

                "overall_similarity": 0.0,

                "matched_percentage": 0.0,

                "total_document_words": 0,

                "matched_document_words": 0,
            }

        # ========================================================
        # COMPARE
        # ========================================================

        matches = self.compare_with_sources(
            passages,
            source_passages
        )

        matches = self.remove_duplicate_matches(
            matches
        )

        # ========================================================
        # CALCULATE TOTAL DOCUMENT WORDS
        # ========================================================

        total_words = 0

        for passage in passages:

            total_words += len(
                self.tokenize(
                    passage["text"]
                )
            )

        # ========================================================
        # CALCULATE MATCHED WORDS
        # ========================================================

        matched_words = 0

        for match in matches:

            passage_words = len(
                self.tokenize(
                    match.get(
                        "text",
                        ""
                    )
                )
            )

            matched_words += min(
                match.get(
                    "longest_match_words",
                    0
                ),

                passage_words
            )

        # ========================================================
        # MATCHED CONTENT PERCENTAGE
        # ========================================================

        if total_words > 0:

            matched_percentage = (
                matched_words
                /
                total_words
                *
                100
            )

        else:

            matched_percentage = 0.0

        matched_percentage = min(
            matched_percentage,
            100.0
        )

        # ========================================================
        # OVERALL SIMILARITY
        # ========================================================

        if matches:

            strengths = [
                match.get(
                    "similarity",
                    0
                )
                for match in matches
            ]

            average_strength = (
                sum(strengths)
                /
                len(strengths)
            )

            overall_similarity = (
                matched_percentage * 0.80
                +
                average_strength
                * 100
                * 0.20
            )

        else:

            overall_similarity = 0.0

        overall_similarity = min(
            overall_similarity,
            100.0
        )

        # ========================================================
        # FINAL RESULT
        # ========================================================

        return {
            "total_sentences": len(
                sentences
            ),

            "total_passages": len(
                passages
            ),

            "matches": matches,

            "overall_similarity": float(
                overall_similarity
            ),

            "matched_percentage": float(
                matched_percentage
            ),

            "total_document_words": (
                total_words
            ),

            "matched_document_words": (
                matched_words
            ),
        }

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(
        self,
        result
    ):
        matches = result.get(
            "matches",
            []
        )

        high = sum(
            1
            for match in matches
            if match.get(
                "level"
            ) == "high"
        )

        moderate = sum(
            1
            for match in matches
            if match.get(
                "level"
            ) == "moderate"
        )

        return {
            "overall_similarity": (
                result.get(
                    "overall_similarity",
                    0
                )
            ),

            "matched_percentage": (
                result.get(
                    "matched_percentage",
                    0
                )
            ),

            "high_similarity": high,

            "moderate_similarity": moderate,

            "total_matches": len(
                matches
            ),

            "total_sentences": (
                result.get(
                    "total_sentences",
                    0
                )
            ),

            "total_passages": (
                result.get(
                    "total_passages",
                    0
                )
            ),
        }