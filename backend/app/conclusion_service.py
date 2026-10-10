# ============================================================
# US07：可解释岗位匹配结论
#
# 根据 US04、US05、US06 的实际结果生成综合结论。
# 使用透明的规则评分，不额外调用 LLM 生成分数。
# ============================================================

from typing import Any


# 各维度权重
DIMENSION_WEIGHTS = {
    "hard_conditions": 0.40,
    "skills": 0.30,
    "experiences": 0.30,
}


# ============================================================
# 通用工具
# ============================================================

def _as_list(value: Any) -> list:
    """将输入统一转换为列表。"""

    if isinstance(value, list):
        return value

    if value is None or value == "":
        return []

    return [value]


def _text(value: Any) -> str:
    """将值转换为适合展示的文本。"""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        return "；".join(
            f"{key}: {val}"
            for key, val in value.items()
            if val not in (None, "", [], {})
        )

    if isinstance(value, list):
        return "；".join(
            _text(item)
            for item in value
            if _text(item)
        )

    return str(value)


def _field(
    item: dict,
    candidates: list[str],
    default: str = "",
) -> str:
    """按照候选字段名获取第一个非空值。"""

    for key in candidates:
        value = item.get(key)

        if value not in (None, "", [], {}):
            return _text(value)

    return default


# ============================================================
# 状态分类
# ============================================================

def _classify_status(status: str) -> str:
    """
    将不同服务可能使用的状态统一分类。

    返回：
    matched / partial / unmatched / unknown
    """

    value = status.strip().lower()

    if not value:
        return "unknown"

    # 必须先识别否定状态。
    # 否则“不满足”可能被错误识别成“满足”。
    negative_terms = (
        "不满足",
        "不匹配",
        "不符合",
        "未通过",
        "未达到",
        "不具备",
        "没有相关",
        "未发现相关",
        "明显缺口",
        "不建议",
    )

    if any(
        term in value
        for term in negative_terms
    ):
        return "unmatched"

    partial_terms = (
        "部分匹配",
        "部分满足",
        "部分符合",
        "匹配度一般",
        "整体匹配度一般",
        "存在一定缺口",
    )

    if any(
        term in value
        for term in partial_terms
    ):
        return "partial"

    unknown_terms = (
        "信息不足",
        "无法判断",
        "无法确定",
        "未知",
        "待确认",
        "未提供",
        "证据不足",
    )

    if any(
        term in value
        for term in unknown_terms
    ):
        return "unknown"

    positive_terms = (
        "满足",
        "匹配",
        "符合",
        "通过",
        "具备",
        "较高",
    )

    if any(
        term in value
        for term in positive_terms
    ):
        return "matched"

    return "unknown"


# ============================================================
# 提取证据
# ============================================================

def _extract_evidence(item: dict) -> str:
    """整理匹配服务返回的证据。"""

    parts = []

    experiences = item.get(
        "matched_experiences",
        [],
    )

    experience_text = _text(experiences)

    if experience_text:
        parts.append(
            f"对应经历：{experience_text}"
        )

    evidence = _field(
        item,
        [
            "evidence",
            "match_evidence",
            "matching_evidence",
        ],
    )

    if evidence:
        parts.append(
            f"匹配依据：{evidence}"
        )

    explanation = _field(
        item,
        [
            "explanation",
            "reason",
            "analysis",
        ],
    )

    if explanation:
        parts.append(
            f"分析说明：{explanation}"
        )

    return "；".join(parts)


# ============================================================
# 标准化分析条目
# ============================================================

def _normalize_rows(
    items: Any,
    dimension_label: str,
) -> list[dict]:
    """将不同模块的输出统一为相同的证据条目格式。"""

    rows = []

    for index, item in enumerate(
        _as_list(items),
        start=1,
    ):

        if isinstance(item, dict):

            title = _field(
                item,
                [
                    "condition",
                    "work_item",
                    "skill",
                    "name",
                    "requirement",
                    "title",
                ],
                default=f"分析条目 {index}",
            )

            status = _field(
                item,
                [
                    "status",
                    "match_status",
                    "result",
                ],
                default="信息不足",
            )

            explanation = _field(
                item,
                [
                    "explanation",
                    "reason",
                    "analysis",
                ],
            )

            evidence = _extract_evidence(
                item
            )

        else:

            title = _text(item) or (
                f"分析条目 {index}"
            )

            status = "信息不足"
            explanation = ""
            evidence = ""


        rows.append({
            "dimension": dimension_label,
            "title": title,
            "status": status,
            "classification": (
                _classify_status(status)
            ),
            "evidence": evidence,
            "explanation": explanation,
        })

    return rows


# ============================================================
# 单一维度分析
# ============================================================

def _analyze_dimension(
    label: str,
    items: Any,
    overall_status: str = "",
) -> tuple[dict, list[dict]]:
    """
    汇总某个维度的匹配情况。

    匹配：1 分
    部分匹配：0.5 分
    不匹配：0 分
    信息不足：不直接计为失败，并单独统计。
    """

    rows = _normalize_rows(
        items,
        label,
    )

    # 如果没有明细，才考虑使用模块整体状态。
    if not rows and overall_status:

        rows = _normalize_rows(
            [overall_status],
            label,
        )

        if rows:
            rows[0]["title"] = (
                f"{label}整体结果"
            )
            rows[0]["status"] = (
                overall_status
            )
            rows[0]["classification"] = (
                _classify_status(
                    overall_status
                )
            )


    matched = sum(
        row["classification"] == "matched"
        for row in rows
    )

    partial = sum(
        row["classification"] == "partial"
        for row in rows
    )

    unmatched = sum(
        row["classification"] == "unmatched"
        for row in rows
    )

    unknown = sum(
        row["classification"] == "unknown"
        for row in rows
    )

    total = len(rows)

    known = (
        matched
        + partial
        + unmatched
    )


    # 只在有可判断结果的条目上计算得分。
    score = None

    if known > 0:

        score = round(
            (
                matched
                + partial * 0.5
            )
            / known
            * 100
        )


    coverage = 0

    if total > 0:

        coverage = round(
            known / total * 100
        )


    if total == 0:

        summary = (
            f"{label}暂无可用分析结果。"
        )

    else:

        summary = (
            f"共 {total} 项："
            f"匹配 {matched} 项，"
            f"部分匹配 {partial} 项，"
            f"不匹配 {unmatched} 项，"
            f"信息不足 {unknown} 项。"
        )


    result = {
        "label": label,
        "score": score,
        "total_count": total,
        "matched_count": matched,
        "partial_count": partial,
        "unmatched_count": unmatched,
        "unknown_count": unknown,
        "coverage": coverage,
        "summary": summary,
    }

    return result, rows


# ============================================================
# 生成综合结论
# ============================================================

def build_explainable_conclusion(
    job: dict,
    hard_match: dict,
    skill_match: dict,
    experience_match: dict,
) -> dict:
    """
    综合 US04、US05、US06 的结果，生成可解释结论。

    不重新调用 LLM，不创建用户原本没有的经历证据。
    """

    # --------------------------------------------------------
    # 1. 分别分析三个维度
    # --------------------------------------------------------

    hard_dimension, hard_rows = (
        _analyze_dimension(
            "硬性条件",
            hard_match.get(
                "conditions",
                [],
            ),
            hard_match.get(
                "overall_status",
                "",
            ),
        )
    )

    skill_dimension, skill_rows = (
        _analyze_dimension(
            "技能匹配",
            skill_match.get(
                "skills",
                [],
            ),
            skill_match.get(
                "overall_status",
                "",
            ),
        )
    )

    experience_dimension, experience_rows = (
        _analyze_dimension(
            "项目 / 实习经历",
            experience_match.get(
                "work_items",
                [],
            ),
            experience_match.get(
                "overall_status",
                "",
            ),
        )
    )


    dimensions = {
        "hard_conditions": {
            **hard_dimension,
            "label": "硬性条件",
        },
        "skills": {
            **skill_dimension,
            "label": "技能匹配",
        },
        "experiences": {
            **experience_dimension,
            "label": "项目 / 实习经历",
        },
    }


    all_rows = (
        hard_rows
        + skill_rows
        + experience_rows
    )


    # --------------------------------------------------------
    # 2. 计算综合分数
    # --------------------------------------------------------

    weighted_score = 0.0
    available_weight = 0.0

    weighted_coverage = 0.0
    coverage_weight = 0.0

    for key, weight in (
        DIMENSION_WEIGHTS.items()
    ):

        dimension = dimensions[key]

        score = dimension["score"]

        if score is not None:

            weighted_score += (
                score * weight
            )

            available_weight += weight


        if dimension["total_count"] > 0:

            weighted_coverage += (
                dimension["coverage"]
                * weight
            )

            coverage_weight += weight


    overall_score = None

    if available_weight > 0:

        overall_score = round(
            weighted_score
            / available_weight
        )


    evidence_coverage = 0

    if coverage_weight > 0:

        evidence_coverage = round(
            weighted_coverage
            / coverage_weight
        )


    # --------------------------------------------------------
    # 3. 判断是否存在硬性条件风险
    # --------------------------------------------------------

    hard_blocker = (
        hard_dimension["unmatched_count"] > 0
    )


    # --------------------------------------------------------
    # 4. 输出整体匹配结论
    # --------------------------------------------------------

    if hard_blocker:

        overall_status = (
            "存在硬性条件风险"
        )

    elif (
        overall_score is None
        or evidence_coverage < 35
    ):

        overall_status = (
            "信息不足，暂无法判断"
        )

    elif (
        overall_score >= 80
        and evidence_coverage >= 60
    ):

        overall_status = (
            "整体匹配度较高"
        )

    elif overall_score >= 55:

        overall_status = (
            "整体匹配度一般"
        )

    else:

        overall_status = (
            "整体匹配度较低"
        )


    # --------------------------------------------------------
    # 5. 提炼优势与风险
    # --------------------------------------------------------

    highlights = []

    risks = []

    for row in all_rows:

        item = {
            "dimension": row["dimension"],
            "title": row["title"],
            "status": row["status"],
            "evidence": row["evidence"],
            "explanation": row["explanation"],
        }

        classification = row[
            "classification"
        ]

        if classification == "matched":

            highlights.append(item)

        else:

            risks.append(item)


    # --------------------------------------------------------
    # 6. 生成可执行建议
    # --------------------------------------------------------

    recommendations = []


    if (
        hard_dimension["unmatched_count"] > 0
    ):

        recommendations.append(
            "优先核实未满足的硬性条件。"
            "如果属于不可协商的招聘要求，"
            "应谨慎评估是否继续投递。"
        )


    for risk in risks:

        title = risk["title"]
        status = risk["status"]
        dimension = risk["dimension"]

        if "技能" in dimension:

            recommendations.append(
                f"针对“{title}”，"
                "核实自身掌握程度，并通过"
                "真实项目、练习或学习记录"
                "补充相关能力证据。"
            )

        elif "经历" in dimension:

            recommendations.append(
                f"针对“{title}”，"
                "检查现有项目和实习描述中"
                "是否有真实、明确的对应经历；"
                "如果确实没有，应考虑通过实际项目"
                "补齐相关经验，不要虚构经历。"
            )

        elif (
            "硬性条件" in dimension
            and (
                "信息不足" in status
                or "未知" in status
            )
        ):

            recommendations.append(
                f"核实“{title}”的具体要求，"
                "并完善个人画像中的相关信息。"
            )


    if evidence_coverage < 70:

        recommendations.append(
            "部分判断依据不足。"
            "建议完善个人画像中的技能、"
            "项目内容和实习职责，"
            "以减少信息不足的匹配结果。"
        )


    if not recommendations:

        recommendations.append(
            "当前未发现明显的匹配缺口。"
            "投递前仍应核对岗位原文，"
            "并确保简历中的项目描述真实准确。"
        )


    # 去重并保持原有顺序
    recommendations = list(
        dict.fromkeys(
            recommendations
        )
    )


    # --------------------------------------------------------
    # 7. 形成逐维度解释
    # --------------------------------------------------------

    reasoning = [
        (
            "硬性条件："
            + hard_dimension["summary"]
        ),
        (
            "技能匹配："
            + skill_dimension["summary"]
        ),
        (
            "项目 / 实习经历："
            + experience_dimension["summary"]
        ),
    ]


    if hard_blocker:

        reasoning.append(
            "当前存在硬性条件不满足项，"
            "因此即使其他维度有匹配优势，"
            "也需要优先处理该风险。"
        )


    # --------------------------------------------------------
    # 8. 汇总文字
    # --------------------------------------------------------

    position = _text(
        job.get("position")
    ) or "当前岗位"

    if overall_score is None:

        score_text = (
            "目前缺少足够的可判断证据，"
            "暂不计算综合分数"
        )

    else:

        score_text = (
            f"当前规则评分为 "
            f"{overall_score}/100"
        )


    summary = (
        f"针对“{position}”，"
        f"{overall_status}。"
        f"{score_text}。"
        f"证据覆盖率为 "
        f"{evidence_coverage}%。"
        "该分数是用于解释和比较的项目内规则评分，"
        "不是获得面试或录用的概率。"
    )


    # --------------------------------------------------------
    # 9. 返回结构化结论
    # --------------------------------------------------------

    return {
        "overall_status": overall_status,
        "overall_score": overall_score,
        "evidence_coverage": evidence_coverage,
        "dimensions": dimensions,
        "highlights": highlights[:10],
        "risks": risks[:10],
        "recommendations": recommendations[:10],
        "reasoning": reasoning,
        "summary": summary,
    }