from pathlib import Path
import re

import fitz


class PDFHighlighter:

    def __init__(self):

        pass


    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def normalize(
        text
    ):

        if not text:

            return ""

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    # ========================================================
    # NORMALIZE WORD
    # ========================================================

    @staticmethod
    def normalize_word(
        word
    ):

        if not word:

            return ""

        word = word.lower()

        word = re.sub(
            r"[^\w]",
            "",
            word
        )

        return word


    # ========================================================
    # EXACT SEARCH
    # ========================================================

    def find_exact(
        self,
        page,
        text
    ):

        if not text:

            return []


        clean_text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()


        if len(clean_text) < 5:

            return []


        try:

            quads = page.search_for(
                clean_text,
                quads=True
            )


            if quads:

                return quads

        except Exception:

            pass


        return []


    # ========================================================
    # WORD SEQUENCE SEARCH
    # ========================================================

    def find_word_sequences(
        self,
        page,
        text
    ):

        target_words = re.findall(
            r"\b\w+\b",
            text.lower()
        )


        target_words = [
            self.normalize_word(
                word
            )
            for word in target_words
        ]


        target_words = [
            word
            for word in target_words
            if word
        ]


        if len(target_words) < 3:

            return []


        page_words = page.get_text(
            "words"
        )


        if not page_words:

            return []


        page_tokens = []


        for word_data in page_words:

            word = self.normalize_word(
                word_data[4]
            )


            if not word:

                continue


            page_tokens.append(
                {
                    "word": word,
                    "rect": fitz.Rect(
                        word_data[:4]
                    )
                }
            )


        page_sequence = [
            item["word"]
            for item in page_tokens
        ]


        matches = []


        # ----------------------------------------------------
        # Search largest chunks first.
        # ----------------------------------------------------

        chunk_sizes = [
            20,
            16,
            12,
            10,
            8,
            6,
            5,
            4,
            3
        ]


        for chunk_size in chunk_sizes:

            if len(target_words) < chunk_size:

                continue


            for target_start in range(
                0,
                len(target_words)
                - chunk_size
                + 1
            ):

                target_chunk = (
                    target_words[
                        target_start:
                        target_start
                        + chunk_size
                    ]
                )


                for page_start in range(
                    0,
                    len(page_sequence)
                    - chunk_size
                    + 1
                ):

                    page_chunk = (
                        page_sequence[
                            page_start:
                            page_start
                            + chunk_size
                        ]
                    )


                    if (
                        page_chunk
                        == target_chunk
                    ):

                        matches.append(
                            page_tokens[
                                page_start:
                                page_start
                                + chunk_size
                            ]
                        )


                        # One good exact run is enough.
                        break


                if matches:

                    break


            if matches:

                break


        return matches


    # ========================================================
    # MERGE RECTANGLES BY LINE
    # ========================================================

    @staticmethod
    def merge_rectangles(
        word_items
    ):

        if not word_items:

            return []


        lines = []


        for item in word_items:

            rect = item["rect"]

            placed = False


            for line in lines:

                reference = line[0]


                if abs(
                    rect.y0
                    - reference.y0
                ) < 8:

                    line.append(
                        rect
                    )

                    placed = True

                    break


            if not placed:

                lines.append(
                    [rect]
                )


        rectangles = []


        for line in lines:

            x0 = min(
                rect.x0
                for rect in line
            )


            y0 = min(
                rect.y0
                for rect in line
            )


            x1 = max(
                rect.x1
                for rect in line
            )


            y1 = max(
                rect.y1
                for rect in line
            )


            rectangles.append(
                fitz.Rect(
                    x0,
                    y0,
                    x1,
                    y1
                )
            )


        return rectangles


    # ========================================================
    # ADD HIGHLIGHT
    # ========================================================

    @staticmethod
    def add_highlight(
        page,
        rect,
        highlight_type
    ):

        annot = (
            page.add_highlight_annot(
                rect
            )
        )


        # ----------------------------------------------------
        # RED = strong plagiarism
        # ----------------------------------------------------

        if (
            highlight_type
            == "plagiarism_high"
        ):

            annot.set_colors(
                stroke=(
                    1,
                    0,
                    0
                )
            )


        # ----------------------------------------------------
        # YELLOW = moderate plagiarism
        # ----------------------------------------------------

        elif (
            highlight_type
            == "plagiarism_moderate"
        ):

            annot.set_colors(
                stroke=(
                    1,
                    0.75,
                    0
                )
            )


        # ----------------------------------------------------
        # BLUE = AI likelihood
        # ----------------------------------------------------

        elif (
            highlight_type
            == "ai"
        ):

            annot.set_colors(
                stroke=(
                    0,
                    0.35,
                    1
                )
            )


        annot.update()


    # ========================================================
    # HIGHLIGHT MATCH
    # ========================================================

    def highlight_match(
        self,
        page,
        match,
        highlight_type
    ):

        text = match.get(
            "text",
            ""
        )


        if not text:

            return 0


        count = 0


        # ----------------------------------------------------
        # 1. Exact complete passage
        # ----------------------------------------------------

        quads = self.find_exact(
            page,
            text
        )


        if quads:

            for quad in quads:

                self.add_highlight(
                    page,
                    quad,
                    highlight_type
                )

                count += 1


            return count


        # ----------------------------------------------------
        # 2. Exact word sequence
        # ----------------------------------------------------

        runs = (
            self.find_word_sequences(
                page,
                text
            )
        )


        for run in runs:

            rectangles = (
                self.merge_rectangles(
                    run
                )
            )


            for rect in rectangles:

                self.add_highlight(
                    page,
                    rect,
                    highlight_type
                )

                count += 1


        # ----------------------------------------------------
        # 3. Fallback individual meaningful words
        # ----------------------------------------------------

        if count == 0:

            target_words = set(
                self.normalize_word(
                    word
                )
                for word in re.findall(
                    r"\b\w+\b",
                    text.lower()
                )
            )


            ignored = {
                "the",
                "and",
                "for",
                "with",
                "that",
                "this",
                "from",
                "are",
                "was",
                "were",
                "have",
                "has",
                "been",
                "into",
                "using",
                "used"
            }


            page_words = page.get_text(
                "words"
            )


            for word_data in page_words:

                page_word = (
                    self.normalize_word(
                        word_data[4]
                    )
                )


                if (
                    page_word
                    in target_words
                    and page_word
                    not in ignored
                ):

                    rect = fitz.Rect(
                        word_data[:4]
                    )


                    self.add_highlight(
                        page,
                        rect,
                        highlight_type
                    )


                    count += 1


        return count


    # ========================================================
    # CREATE HIGHLIGHTED PDF
    # ========================================================

    def create_highlighted_pdf(
        self,
        input_pdf,
        output_pdf,
        matches=None,
        ai_matches=None
    ):

        input_pdf = Path(
            input_pdf
        )

        output_pdf = Path(
            output_pdf
        )


        if not input_pdf.exists():

            raise FileNotFoundError(
                f"Input PDF not found: "
                f"{input_pdf}"
            )


        matches = (
            matches
            or []
        )


        ai_matches = (
            ai_matches
            or []
        )


        doc = fitz.open(
            str(input_pdf)
        )


        plagiarism_highlights = 0

        ai_highlights = 0


        # ====================================================
        # PLAGIARISM
        # ====================================================

        for match in matches:

            page_number = match.get(
                "page"
            )


            if not page_number:

                continue


            try:

                page_index = (
                    int(page_number)
                    - 1
                )

            except Exception:

                continue


            if (
                page_index < 0
                or page_index >= len(doc)
            ):

                continue


            page = doc[
                page_index
            ]


            if (
                match.get("level")
                == "high"
            ):

                highlight_type = (
                    "plagiarism_high"
                )

            else:

                highlight_type = (
                    "plagiarism_moderate"
                )


            plagiarism_highlights += (
                self.highlight_match(
                    page,
                    match,
                    highlight_type
                )
            )


        # ====================================================
        # AI
        # ====================================================

        for match in ai_matches:

            page_number = match.get(
                "page"
            )


            if not page_number:

                continue


            try:

                page_index = (
                    int(page_number)
                    - 1
                )

            except Exception:

                continue


            if (
                page_index < 0
                or page_index >= len(doc)
            ):

                continue


            page = doc[
                page_index
            ]


            ai_highlights += (
                self.highlight_match(
                    page,
                    match,
                    "ai"
                )
            )


        # ====================================================
        # SAVE
        # ====================================================

        if output_pdf.exists():

            output_pdf.unlink()


        doc.save(
            str(output_pdf),
            garbage=4,
            deflate=True
        )


        doc.close()


        return {
            "output_path": str(
                output_pdf
            ),

            "highlights": (
                plagiarism_highlights
                + ai_highlights
            ),

            "plagiarism_highlights": (
                plagiarism_highlights
            ),

            "ai_highlights": (
                ai_highlights
            ),

            "matches_processed": len(
                matches
            ),

            "ai_matches_processed": len(
                ai_matches
            )
        }