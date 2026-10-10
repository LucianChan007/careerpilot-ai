# ============================================================
# US07：可解释岗位匹配结论的数据结构
# ============================================================

EXPLAINABLE_CONCLUSION_SCHEMA = {
    "overall_status": "",
    "overall_score": None,
    "evidence_coverage": 0,
    "dimensions": {
        "hard_conditions": {
            "label": "硬性条件",
            "score": None,
            "total_count": 0,
            "matched_count": 0,
            "partial_count": 0,
            "unmatched_count": 0,
            "unknown_count": 0,
            "coverage": 0,
            "summary": "",
        },
        "skills": {
            "label": "技能匹配",
            "score": None,
            "total_count": 0,
            "matched_count": 0,
            "partial_count": 0,
            "unmatched_count": 0,
            "unknown_count": 0,
            "coverage": 0,
            "summary": "",
        },
        "experiences": {
            "label": "项目 / 实习经历",
            "score": None,
            "total_count": 0,
            "matched_count": 0,
            "partial_count": 0,
            "unmatched_count": 0,
            "unknown_count": 0,
            "coverage": 0,
            "summary": "",
        },
    },
    "highlights": [],
    "risks": [],
    "recommendations": [],
    "reasoning": [],
    "summary": "",
}