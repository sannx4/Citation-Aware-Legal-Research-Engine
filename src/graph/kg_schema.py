NODE_TYPES = {
    "CASE",
    "COURT",
    "JUDGE",
    "BENCH",
    "ARTICLE",
    "SECTION",
    "ACT",
    "LEGAL_CONCEPT",
    "ISSUE",
    "HOLDING",
    "OUTCOME",
    "CITATION_TREATMENT",
}

EDGE_TYPES = {
    "CASE_CITES_CASE",
    "CASE_MENTIONS_ARTICLE",
    "CASE_INTERPRETS_SECTION",
    "CASE_HAS_CONCEPT",
    "CASE_DECIDED_BY_COURT",
    "CASE_DECIDED_BY_JUDGE",
    "CASE_HAS_OUTCOME",
    "CASE_HAS_HOLDING",
    "CASE_HAS_CITATION_TREATMENT",
    "CASE_FOLLOWS_CASE",
    "CASE_OVERRULES_CASE",
    "CASE_DISTINGUISHES_CASE",
    "ARTICLE_BELONGS_TO_ACT",
    "SECTION_BELONGS_TO_ACT",
}


def make_safe_id(value: str) -> str:
    value = str(value).strip().upper()
    value = value.replace(" ", "_")
    value = value.replace(".", "")
    value = value.replace(",", "")
    value = value.replace("-", "_")
    value = value.replace("/", "_")
    value = value.replace("(", "")
    value = value.replace(")", "")
    return value


def validate_node_type(node_type: str) -> None:
    if node_type not in NODE_TYPES:
        raise ValueError(f"Invalid node type: {node_type}")


def validate_edge_type(edge_type: str) -> None:
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Invalid edge type: {edge_type}")
