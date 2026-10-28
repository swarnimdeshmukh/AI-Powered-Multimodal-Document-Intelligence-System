from src.ingestion.pdf_extractor import PDFExtractor

PDF_PATH = "data/annual_report.pdf"

def main():
    extractor = PDFExtractor(PDF_PATH)
    result = extractor.extract()

    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Text blocks : {len(result['text'])}")
    print(f"Images      : {len(result['images'])}")
    print(f"Tables      : {len(result['tables'])}")

    if result["text"]:
        print("\nSample text:")
        print(result["text"][0]["content"][:500])

    if result["images"]:
        print("\nSample image:")
        print(result["images"][0]["image_path"])

    if result["tables"]:
        print("\nSample table:")
        for row in result["tables"][0]["content"][:5]:
            print(row)

if __name__ == "__main__":
    main()
