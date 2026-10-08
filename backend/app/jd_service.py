import json
import os

import httpx

from app.jd_schema import JOB_SCHEMA


def build_jd_prompt(jd_text: str) -> str:
    schema = json.dumps(
        JOB_SCHEMA,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是大学生求职岗位信息结构化助手。

任务：
从下面的岗位 JD 原文中提取结构化岗位信息。

严格要求：

1. 只能提取 JD 中明确出现的信息。
2. 不得虚构公司、岗位、技能、学历、专业、经验、地点、薪资或其他要求。
3. 无法判断的信息使用空字符串或空数组。
4. 不要根据常识自行补充岗位要求。
5. 岗位职责必须尽量保留原始语义。
6. required_skills 只放岗位明确要求或明确提及的技能。
7. salary 专门用于存放薪资、薪酬、日薪、月薪、时薪、实习补贴等信息。
8. 薪资信息绝对不能放入 other_requirements。
9. salary 不属于岗位硬性条件，不参与后续硬性条件匹配。
10. other_requirements 只放无法归入学历、专业、技能、经验、地点、实习周期等字段的明确要求。
11. 输出必须是合法 JSON。
12. 不要输出 Markdown。
13. 输出结构必须与 Schema 完全一致。

Schema：
{schema}

JD 原文：
---BEGIN JOB DESCRIPTION---
{jd_text}
---END JOB DESCRIPTION---
""".strip()


async def build_job(jd_text: str) -> dict:
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
        return {
            **JOB_SCHEMA,
            "_mode": "parser_only",
            "_message": "未配置 LLM API。",
        }

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "严格按照指定 JSON Schema 输出结构化岗位信息。"
                ),
            },
            {
                "role": "user",
                "content": build_jd_prompt(jd_text),
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

    # 防止模型遗漏新增字段
    for key, default_value in JOB_SCHEMA.items():
        if key not in result:
            result[key] = default_value

    return result