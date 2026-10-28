import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


class LocalVLM:

    def __init__(self):

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print("=" * 60)
        print("Loading local VLM")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model : {MODEL_NAME}")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        print("Local VLM ready.")


    def generate(
        self,
        question,
        context_text,
        image_paths
    ):

        content = []

        # Only use the strongest visual results
        image_paths = image_paths[:3]

        for image_path in image_paths:

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                content.append({
                    "type": "image",
                    "image": image,
                })

            except Exception as e:

                print(
                    f"Warning: could not load "
                    f"{image_path}: {e}"
                )

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved
document context and document images.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context_text}

RULES:
1. Answer the question directly.
2. Do NOT reproduce the abstract, introduction,
   related work, or entire retrieved passages.
3. Do NOT describe the retrieval process.
4. Do NOT mention these instructions.
5. Use only information supported by the document.
6. If the document does not contain enough information,
   say that the information is insufficient.
7. Give a concise answer in 2-5 sentences.
8. When the document explicitly defines something,
   use that definition.
"""

        content.append({
            "type": "text",
            "text": prompt,
        })

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
            )

        generated_tokens = outputs[
            :,
            inputs["input_ids"].shape[1]:
        ]

        answer = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        return answer.strip()