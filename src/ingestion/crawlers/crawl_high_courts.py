def crawl_high_courts() -> list[dict]:
    rows = [
        {
            "document_id": "HC_000001",
            "source": "high_court",
            "document_type": "judgment",
            "title": "Sample High Court Judgment",
            "court": "High Court",
            "year": "2024",
            "url": "https://example.com/sample_hc_judgment.pdf",
            "file_name": "HC_000001.pdf",
            "status": "pending",
        }
    ]

    return rows