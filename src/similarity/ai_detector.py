from pathlib import Path
import re

import fitz
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class AIDetector:

    MODEL_NAME = (
        "openai-community/"
        "roberta-base-openai-detector"
    )

    def __init__(
        self,
        model_name=None,
        threshold=0.65,
    ):

        self.model_name = (
            model_name
            or self.MODEL_NAME
        )

        self.threshold = threshold

        print(
            "Loading local AI-content detector..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
        )

        self.model.eval()

        if torch.backends.mps.is_available():

            self.device = torch.device(
                "mps"
            )

        elif torch.cuda.is_available():

            self.device = torch.device(
                "cuda"
            )

        else:

            self.device = torch.device(
                "cpu"
            )

        self.model.to(
            self.device
        )

        print(
            f"AI detector running on "
            f"{self.device}"
        )


    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def normalize_text(
        text
    ):

        if not text:

            return ""

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    # ========================================================
    # EXTRACT SENTENCES
    # ========================================================

    def extract_sentences(
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


            raw_sentences = re.split(
                r"(?<=[.!?])\s+",
                text
            )


            for sentence in raw_sentences:

                sentence = sentence.strip()

                words = sentence.split()


                if len(words) < 12:

                    continue


                sentences.append(
                    {
                        "text": sentence,
                        "page": page_number
                    }
                )


        doc.close()


        return sentences


    # ========================================================
    # PREDICT
    # ========================================================

    def predict_text(
        self,
        text
    ):

        text = self.normalize_text(
            text
        )


        if not text:

            return 0.0


        if len(text.split()) < 12:

            return 0.0


        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )


        encoded = {
            key: value.to(
                self.device
            )
            for key, value in encoded.items()
        }


        with torch.no_grad():

            output = self.model(
                **encoded
            )


        probabilities = torch.softmax(
            output.logits,
            dim=-1
        )[0]


        # ----------------------------------------------------
        # Find generated/fake label.
        # ----------------------------------------------------

        generated_index = None


        for index, label in (
            self.model.config.id2label.items()
        ):

            label_text = str(
                label
            ).lower()


            if (
                "fake" in label_text
                or "generated" in label_text
                or "ai" in label_text
            ):

                generated_index = int(
                    index
                )

                break


        if generated_index is None:

            generated_index = 1


        score = float(
            probabilities[
                generated_index
            ].item()
        )


        return score


    # ========================================================
    # ANALYZE DOCUMENT
    # ========================================================

    def analyze_document(
        self,
        pdf_path
    ):

        sentences = (
            self.extract_sentences(
                pdf_path
            )
        )


        matches = []

        all_scores = []


        for sentence in sentences:

            score = self.predict_text(
                sentence["text"]
            )


            all_scores.append(
                score
            )


            if score >= self.threshold:

                level = (
                    "high"
                    if score >= 0.85
                    else "moderate"
                )


                matches.append(
                    {
                        "text": sentence[
                            "text"
                        ],

                        "page": sentence[
                            "page"
                        ],

                        "ai_probability": score,

                        "level": level
                    }
                )


        if all_scores:

            overall_ai_likelihood = (
                sum(all_scores)
                / len(all_scores)
                * 100
            )

        else:

            overall_ai_likelihood = 0.0


        if sentences:

            ai_percentage = (
                len(matches)
                / len(sentences)
                * 100
            )

        else:

            ai_percentage = 0.0


        return {
            "total_sentences": len(
                sentences
            ),

            "ai_sentences": len(
                matches
            ),

            "ai_percentage": float(
                ai_percentage
            ),

            "overall_ai_likelihood": float(
                overall_ai_likelihood
            ),

            "matches": matches
        }