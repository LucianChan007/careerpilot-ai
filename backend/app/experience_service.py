import json
import os

import httpx

from app.experience_schema import EXPERIENCE_MATCH_SCHEMA


def build_experience_match_prompt(
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

    responsibilities = job.get(
        "responsibilities",
        [],
    )

    return f"""
你是大学生求职经历匹配助手。

任务：

分析岗位实际工作内容，并判断用户过去的项目经历和实习经历
与这些工作内容之间的匹配程度。

用户个人画像：
---BEGIN PROFILE---
{profile_json}
---END PROFILE---

岗位信息：
---BEGIN JOB---
{job_json}
---END JOB---

岗位工作内容：
{json.dumps(responsibilities, ensure_ascii=False, indent=2)}

严格要求：

1. 只分析岗位 responsibilities 中明确出现的工作内容。
2. 不得根据职位名称自行虚构工作内容。
3. 每一项主要工作内容必须单独分析。
4. 每项工作内容必须返回：
   - work_item
   - status
   - matched_experiences
   - evidence
   - explanation
5. status 只能是：
   - 匹配
   - 部分匹配
   - 信息不足
   - 不匹配
6. matched_experiences 必须明确指出匹配的项目或实习经历。
7. 如果匹配来源是项目经历，应明确指出项目名称。
8. 如果匹配来源是实习经历，应明确指出公司、岗位或经历名称。
9. evidence 必须来自用户真实个人画像。
10. 如果用户个人画像中没有相关项目或实习经历，必须明确说明“未发现相关经历”。
11. 不得根据常识推断用户一定做过某项工作。
12. 不得虚构用户没有的项目、实习、技术工作或职责。
13. 即使用户有相同技能，也不能仅因为技能相同就认为工作内容完全匹配。
14. 应结合具体项目内容和实习内容判断是否存在工作内容对应关系。
15. salary 不属于工作内容，不参与本次匹配。
16. 岗位技能本身也不能直接作为工作内容匹配结果，必须关注岗位实际工作内容。
17. 如果工作内容与用户经历存在明显相关但无法完全对应，应返回“部分匹配”。
18. 如果个人画像信息不足以判断，应返回“信息不足”。
19. 如果存在明确相反或完全不相关的经历，可以返回“不匹配”。
20. gaps 只记录明显缺少相关项目或实习证据的工作内容。
21. overall_status 只能是：
   - 经历匹配度高
   - 经历匹配度一般
   - 存在明显经历缺口
   - 信息不足
22. 如果多数工作内容都有明确项目或实习证据，可以返回“经历匹配度高”。
23. 如果部分工作内容有对应经历，部分只有间接证据，可以返回“经历匹配度一般”。
24. 如果多个核心工作内容都没有对应经历，应返回“存在明显经历缺口”。
25. 如果大部分工作内容都无法判断，应返回“信息不足”。
26. summary 应用一句话总结项目/实习经历与岗位工作内容的整体关系。
27. 输出必须是合法 JSON。
28. 不要输出 Markdown。
29. 输出结构必须与 Schema 一致。

输出 Schema：
{json.dumps(EXPERIENCE_MATCH_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


async def match_experiences_to_job(
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

    if (
        not base_url
        or not api_key
        or not model
    ):
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
                    "严格按照指定 JSON Schema 输出项目和经历匹配结果。"
                ),
            },
            {
                "role": "user",
                "content": build_experience_match_prompt(
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
        timeout=60,
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

    for key, default_value in EXPERIENCE_MATCH_SCHEMA.items():

        if key not in result:
            result[key] = default_value

    work_items = result.get(
        "work_items",
        [],
    )

    if not isinstance(
        work_items,
        list,
    ):
        work_items = []

    result["work_item_count"] = len(
        work_items
    )

    result["matched_count"] = sum(
        1
        for item in work_items
        if item.get("status") == "匹配"
    )

    result["partial_count"] = sum(
        1
        for item in work_items
        if item.get("status") == "部分匹配"
    )

    result["unknown_count"] = sum(
        1
        for item in work_items
        if item.get("status") == "信息不足"
    )

    result["unmatched_count"] = sum(
        1
        for item in work_items
        if item.get("status") == "不匹配"
    )

    return result