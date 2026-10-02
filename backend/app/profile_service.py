import json
import os
import httpx
from app.profile_schema import PROFILE_SCHEMA

def build_prompt(resume_text: str) -> str:
    schema = json.dumps(PROFILE_SCHEMA, ensure_ascii=False, indent=2)
    return f"""你是大学生求职信息结构化助手。
从简历原文提取结构化信息。
要求：
1. 只提取明确出现或可直接归纳的信息，不得虚构。
2. 无法判断使用空字符串或空数组。
3. 只能输出合法 JSON，不要 Markdown。
4. 字段结构必须与 Schema 一致。

Schema:
{schema}

简历原文：
---BEGIN RESUME---
{resume_text}
---END RESUME---"""

async def build_profile(resume_text: str) -> dict:
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
    if not all([base_url, api_key, model]):
        return {
            **PROFILE_SCHEMA,
            "_mode": "parser_only",
            "_message": "未配置 LLM API，当前仅完成文本提取。"
        }

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "严格按指定 JSON Schema 输出结构化简历信息。"},
            {"role": "user", "content": build_prompt(resume_text)}
        ]
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM API 调用失败：HTTP {r.status_code} {r.text[:500]}")
    content = r.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError("模型返回的不是合法 JSON。") from e
