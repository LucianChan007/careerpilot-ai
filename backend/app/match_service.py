import json
import os

import httpx

from app.match_schema import MATCH_SCHEMA


def build_match_prompt(
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
你是大学生求职岗位匹配助手。

任务：
根据“用户个人画像”和“岗位结构化信息”，判断用户是否满足岗位明确的硬性条件。

用户个人画像：
---BEGIN PROFILE---
{profile_json}
---END PROFILE---

岗位信息：
---BEGIN JOB---
{job_json}
---END JOB---

匹配规则：

1. 只判断岗位中明确出现的硬性条件。
2. 重点检查：
   - 学历
   - 专业
   - 技能
   - 经验
   - 工作地点
   - 实习周期
3. 每一个条件必须返回：
   - category
   - requirement
   - status
   - evidence
4. status 只能是：
   - 满足
   - 不满足
   - 信息不足
5. 如果个人画像中没有足够证据，不得假设用户满足，应该返回“信息不足”。
6. 可以结合用户的技能、项目和实习经历判断技能和经验是否存在明确证据。
7. salary 只是岗位信息，不属于硬性条件，绝对不能因为薪资而判定岗位匹配失败。
8. salary 必须单独放入 salary 字段。
9. other_requirements 不应直接视为硬性条件，除非其中明确属于岗位硬性要求。
10. 不得虚构用户没有提供的经历、技能或条件。
11. overall_status 只能是：
   - 满足硬性条件
   - 存在硬性不匹配
   - 信息不足
12. 如果存在任何明确“不满足”的硬性条件，overall_status 必须为“存在硬性不匹配”。
13. 如果没有明确不满足，但存在“信息不足”，overall_status 为“信息不足”。
14. 只有所有可判断的硬性条件都满足且不存在信息不足时，才能返回“满足硬性条件”。
15. 输出必须是合法 JSON。
16. 不要输出 Markdown。
17. 输出结构必须与 Schema 一致。

输出 Schema：
{json.dumps(MATCH_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


async def match_profile_to_job(
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
                    "严格按照指定 JSON Schema 输出岗位匹配结果。"
                ),
            },
            {
                "role": "user",
                "content": build_match_prompt(
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
        result = json.loads(content)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "模型返回内容不是合法 JSON。"
        ) from exc

    # 补充默认字段
    for key, default_value in MATCH_SCHEMA.items():
        if key not in result:
            result[key] = default_value

    # 强制确保薪资不是硬性条件
    result["salary"] = {
        "value": job.get(
            "salary",
            "",
        ),
        "is_hard_requirement": False,
        "note": (
            "薪资信息仅作为岗位信息展示，"
            "不参与硬性条件匹配。"
        ),
    }

    return result