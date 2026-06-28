import pandas as pd

def normalize_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def row_matches(query_row, candidate_row, fields):
    for field in fields:
        if field not in query_row.index or field not in candidate_row.index:
            return False
        if normalize_value(query_row[field]) != normalize_value(candidate_row[field]):
            return False

    return True


def detect_error_type(query_row, candidate_row):
    checks = [
        ("articleType", "wrong_articleType"),
        ("subCategory", "wrong_subCategory"),
        ("gender", "wrong_gender"),
        ("masterCategory", "wrong_masterCategory"),
    ]

    for field, error_name in checks:
        if field in query_row.index and field in candidate_row.index:
            if normalize_value(query_row[field]) != normalize_value(candidate_row[field]):
                return error_name
    return "ranking_error"


def build_error_case(query_row, candidate_row, score, query_index, candidate_index):
    return {
        "query_index": int(query_index),
        "top1_index": int(candidate_index),
        "query_image": str(query_row.get("image_path", query_row.get("filename", query_row.get("image", "")))),
        "query_articleType": str(query_row.get("articleType", "")),
        "query_subCategory": str(query_row.get("subCategory", "")),
        "query_gender": str(query_row.get("gender", "")),
        "query_masterCategory": str(query_row.get("masterCategory", "")),
        "top1_image": str(candidate_row.get("image_path", candidate_row.get("filename", candidate_row.get("image", "")))),
        "top1_articleType": str(candidate_row.get("articleType", "")),
        "top1_subCategory": str(candidate_row.get("subCategory", "")),
        "top1_gender": str(candidate_row.get("gender", "")),
        "top1_masterCategory": str(candidate_row.get("masterCategory", "")),
        "similarity_score": float(score),
        "error_type": detect_error_type(query_row, candidate_row),
    }