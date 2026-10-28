"""Evaluation entry point for retrieval quality."""

import json
from pathlib import Path

QUESTIONS = Path(__file__).with_name("questions.json")

def main():
    questions = json.loads(QUESTIONS.read_text())
    print(f"Evaluation questions loaded: {len(questions)}")
    print("Precision@k evaluation will be implemented after retrieval is complete.")

if __name__ == "__main__":
    main()
