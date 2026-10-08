import json
import os

import httpx

from app.skill_schema import SKILL_MATCH_SCHEMA


def build_skill_match_prompt(
    profile: dict,
    job: dict,
) -> str:

    profile_json = json.dumps(
        profile,
        ensure_ascii=False,
        indent=2,
    )

    job_json = json.dumps(
        job,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是大学生求职技能匹配助手。

任务：

分析岗位要求的技能，与用户个人画像中的技能、项目经历和实习经历，
判断用户与岗位技能要求之间的匹配程度。

用户个人画像：
---BEGIN PROFILE---
{profile_json}
---END PROFILE---

岗位信息：
---BEGIN JOB---
{job_json}
---END JOB---

匹配规则：

1. 只分析岗位 required_skills 中明确出现的技能。
2. 不得自行增加岗位技能要求。
3. 每个岗位技能都必须单独分析。
4. 每个技能必须返回：
   - skill
   - status
   - evidence
   - explanation
5. status 只能是：
   - 满足
   - 部分匹配
   - 信息不足
   - 不匹配
6. evidence 必须说明证据来自哪里：
   - 用户个人技能
   - 项目经历
   - 实习经历
   - 或多个来源
7. 如果用户技能列表中明确包含该技能，可以优先判断为“满足”。
8. 如果用户没有直接写该技能，但项目或实习经历明确体现了该技能，可以根据具体经历判断。
9. 如果个人画像没有足够证据，不得猜测用户掌握该技能，应返回“信息不足”。
10. 如果个人画像中存在明确相关但不完全对应的技能，可以判断为“部分匹配”。
11. 不得因为两个技能名字看起来相似就直接判定满足，必须结合语义和具体证据。
12. C/C++ 应作为一个整体技能理解。
13. 如果岗位要求 C/C++，用户明确掌握 C++，可以认为存在直接技能匹配。
14. 如果岗位要求 C++，用户只有 C 的证据，不能直接判定为满足，可判断为部分匹配或信息不足。
15. QT 与 Qt 应视为同一技能。
16. salary 不属于技能，不得进入技能匹配结果。
17. 岗位职责不属于技能要求，不得直接作为技能要求进行匹配。
18. 不得虚构用户不存在的项目、实习、技能或工作经历。
19. gaps 只记录明显不满足或信息不足的技能。
20. recommendations 给出简短的技能补强建议，不得虚构具体课程或经历。
21. overall_status 只能是：
   - 技能匹配度高
   - 技能匹配度一般
   - 存在明显技能缺口
   - 信息不足
22. 如果多数技能满足，且没有明显缺口，可以返回“技能匹配度高”。
23. 如果存在部分匹配或较多信息不足，可以返回“技能匹配度一般”。
24. 如果存在多个明确不匹配技能，应返回“存在明显技能缺口”。
25. 如果大部分技能都无法从个人画像判断，则返回“信息不足”。
26. 输出必须是合法 JSON。
27. 不要输出 Markdown。
28. 输出结构必须与 Schema 一致。

岗位需要匹配的技能：
{json.dumps(job.get("required_skills", []), ensure_ascii=False, indent=2)}

输出 Schema：
{json.dumps(SKILL_MATCH_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


async def match_skills_to_profile(
    profile: dict,
    job: dict,
) -> dict:

    base_url = os.getenv(
        "LLM_BASE_URL",
        "",
    ).rstrip("/")

    api_key = os.getenv(
        "LLM_API_KEY",
        "",
    )

    model = os.getenv(
        "LLM_MODEL",
        "",
    )

    if not base_url or not api_key or not model:
        raise RuntimeError(
            "未配置 LLM API。"
        )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "严格按照指定 JSON Schema 输出技能匹配结果。"
                ),
            },
            {
                "role": "user",
                "content": build_skill_match_prompt(
                    profile,
                    job,
                ),
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(
        timeout=60
    ) as client:

        response = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"LLM API 调用失败：HTTP "
            f"{response.status_code} "
            f"{response.text[:500]}"
        )

    content = (
        response.json()["choices"][0]["message"]["content"]
        .strip()
    )

    if content.startswith("```"):
        content = content.replace(
            "```json",
            "",
        )

        content = content.replace(
            "```",
            "",
        )

        content = content.strip()

    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "模型返回内容不是合法 JSON。"
        ) from exc

    # 补充缺失字段
    for key, default_value in SKILL_MATCH_SCHEMA.items():

        if key not in result:

            result[key] = default_value

    # 以 JD 中实际技能数量为准
    required_skills = job.get(
        "required_skills",
        [],
    )

    result["required_count"] = len(
        required_skills
    )

    # 根据结果重新统计
    skill_results = result.get(
        "skills",
        [],
    )

    result["matched_count"] = sum(
        1
        for item in skill_results
        if item.get("status") == "满足"
    )

    result["partial_count"] = sum(
        1
        for item in skill_results
        if item.get("status") == "部分匹配"
    )

    result["unknown_count"] = sum(
        1
        for item in skill_results
        if item.get("status") == "信息不足"
    )

    result["unmatched_count"] = sum(
        1
        for item in skill_results
        if item.get("status") == "不匹配"
    )

    return result
