def crawl_supreme_court() -> list[dict]:
    rows = [
        {
            "document_id": "SC_000001",
            "source": "supreme_court",
            "document_type": "judgment",
            "title": "Sample Supreme Court Judgment",
            "court": "Supreme Court of India",
            "year": "2024",
            "url": "https://example.com/sample_sc_judgment.pdf",
            "file_name": "SC_000001.pdf",
            "status": "pending",
        }
    ]

    return rows