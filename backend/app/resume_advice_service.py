
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from app.resume_advice_schema import RESUME_ADVICE_SCHEMA


# ============================================================
# 1. 环境配置
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(ENV_PATH)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()


# ============================================================
# 2. 通用数据处理
# ============================================================

def _to_plain(value: Any) -> Any:
    """
    将 Pydantic 模型、字典、列表等数据转换为可 JSON 序列化的结构。
    兼容 Pydantic v1、v2 和普通 Python 对象。
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _to_plain(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item) for item in value]

    # Pydantic v2
    if hasattr(value, "model_dump"):
        try:
            return _to_plain(value.model_dump(mode="json"))
        except TypeError:
            return _to_plain(value.model_dump())

    # Pydantic v1
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _to_plain(value.dict())
        except Exception:
            pass

    # 普通对象
    if hasattr(value, "__dict__"):
        try:
            return _to_plain(vars(value))
        except Exception:
            pass

    return str(value)


def to_text(value: Any) -> str:
    """把任意数据转为适合展示或拼入提示词的字符串。"""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    return json.dumps(
        _to_plain(value),
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def as_list(value: Any) -> list:
    """将输入尽可能统一成列表。"""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        # 尝试解析本身就是 JSON 数组的字符串。
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        return [value]

    return [value]


# ============================================================
# 3. JSON 提取与解析
# ============================================================

def extract_json(content: str) -> dict:
    """
    从模型输出中提取一个完整的 JSON 对象。

    兼容：
    1. 直接返回 JSON；
    2. JSON 外层带 Markdown 代码块；
    3. JSON 前后带少量解释文字；
    4. 部分模型返回中包含 <think>...</think> 内容。

    如果 JSON 不完整或格式错误，则抛出详细异常，
    以便定位模型是否截断或输出了非法结构。
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("简历建议模型返回内容为空。")

    text = content.strip()

    # 移除部分模型可能返回的思考标签。
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # 移除 JSON 外面的 Markdown 代码块标记。
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text).strip()

    # 第一次尝试：整个字符串本身就是 JSON。
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("模型返回了 JSON，但最外层不是对象。")
    except json.JSONDecodeError:
        pass

    # 第二次尝试：从文本中提取第一个完整的 JSON 对象。
    # 采用括号深度计数，并处理字符串内部的花括号。
    start = text.find("{")

    if start == -1:
        raise ValueError(
            "简历建议模型返回内容无法解析为 JSON：未找到左花括号。"
        )

    depth = 0
    in_string = False
    escaped = False
    end = -1

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                end = index + 1
                break

    if end == -1:
        raise ValueError(
            "简历建议模型返回内容无法解析为 JSON："
            "未找到完整闭合的 JSON 对象，模型输出可能被截断。"
        )

    candidate = text[start:end]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "简历建议模型返回内容无法解析为 JSON："
            f"{exc.msg}，位置 {exc.pos}。"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("模型返回的 JSON 最外层必须是对象。")

    return parsed


# ============================================================
# 4. 输出格式标准化
# ============================================================

def normalize_text_list(
    value: Any,
    max_items: int | None = None,
) -> list[str]:
    """将输入标准化为字符串列表。"""
    result: list[str] = []

    for item in as_list(value):
        if item is None:
            continue

        if isinstance(item, dict):
            text = to_text(item)
        elif isinstance(item, (list, tuple)):
            text = to_text(item)
        else:
            text = str(item).strip()

        if text:
            result.append(text)

        if max_items is not None and len(result) >= max_items:
            break

    return result


def normalize_object_list(
    value: Any,
    max_items: int = 8,
) -> list[dict]:
    """
    将列表标准化为对象列表。

    正常情况下模型应返回对象数组。
    如果个别条目意外返回字符串，则转成通用对象，避免整个结果失败。
    """
    result: list[dict] = []

    for item in as_list(value):
        if item is None:
            continue

        if isinstance(item, dict):
            normalized_item = _to_plain(item)
        elif isinstance(item, str):
            item = item.strip()

            if not item:
                continue

            normalized_item = {
                "title": item,
                "recommendation": item,
                "reason": "",
            }
        else:
            normalized_item = {
                "title": to_text(item),
                "recommendation": to_text(item),
                "reason": "",
            }

        result.append(normalized_item)

        if len(result) >= max_items:
            break

    return result


def _first_non_empty(data: dict, keys: list[str]) -> str:
    """按照候选字段顺序获取第一个非空值。"""
    for key in keys:
        value = data.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def normalize_resume_advice(
    result: Any,
    job: Any,
) -> dict:
    """
    统一 US09 返回结构，保证前端所需的主要字段始终存在。
    """
    if not isinstance(result, dict):
        raise ValueError("简历建议结果必须是 JSON 对象。")

    job_data = _to_plain(job)
    if not isinstance(job_data, dict):
        job_data = {}

    # 使用深拷贝，避免修改导入的全局 schema。
    normalized = copy.deepcopy(RESUME_ADVICE_SCHEMA)

    # 目标岗位。
    target_position = _first_non_empty(
        result,
        ["target_position", "position_name", "job_title"],
    )

    if not target_position:
        target_position = _first_non_empty(
            job_data,
            [
                "target_position",
                "position_name",
                "job_title",
                "title",
                "position",
                "name",
            ],
        )

    normalized["target_position"] = target_position

    # 顶层文本字段。
    text_fields = [
        "overall_summary",
        "resume_positioning",
    ]

    for field in text_fields:
        value = result.get(field, "")
        normalized[field] = (
            str(value).strip()
            if value is not None
            else ""
        )

    # 简历摘要建议。
    resume_summary = result.get("resume_summary", {})

    if not isinstance(resume_summary, dict):
        resume_summary = {}

    normalized["resume_summary"] = {
        "current_assessment": str(
            resume_summary.get("current_assessment") or ""
        ).strip(),
        "suggested_version": str(
            resume_summary.get("suggested_version") or ""
        ).strip(),
        "evidence_basis": normalize_text_list(
            resume_summary.get("evidence_basis"),
            max_items=8,
        ),
    }

    # 对象列表字段及输出数量限制。
    object_field_limits = {
        "skills_recommendations": 8,
        "project_recommendations": 5,
        "internship_recommendations": 3,
        "bullet_rewrites": 5,
    }

    for field, limit in object_field_limits.items():
        normalized[field] = normalize_object_list(
            result.get(field, []),
            max_items=limit,
        )

    # 字符串列表字段及输出数量限制。
    text_field_limits = {
        "keywords_to_emphasize": 15,
        "do_not_claim": 8,
        "priority_actions": 6,
        "final_checklist": 8,
    }

    for field, limit in text_field_limits.items():
        normalized[field] = normalize_text_list(
            result.get(field, []),
            max_items=limit,
        )

    return normalized


# ============================================================
# 5. 构造模型提示词
# ============================================================

def build_resume_advice_prompt(
    profile: Any,
    job: Any,
    skill_match: Any,
    experience_match: Any,
) -> str:
    """构建 US09 的提示词。"""

    schema_example = json.dumps(
        RESUME_ADVICE_SCHEMA,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    input_data = {
        "candidate_profile": _to_plain(profile),
        "target_job": _to_plain(job),
        "skill_match_result": _to_plain(skill_match),
        "experience_match_result": _to_plain(experience_match),
    }

    input_json = json.dumps(
        input_data,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return f"""
你是一个严谨的技术岗位简历优化助手，负责根据候选人的真实经历和目标岗位要求，
提出有针对性的简历修改建议。

你的目标不是替候选人编造一份更漂亮的简历，而是帮助候选人更准确地展示已有能力，
突出与岗位相关的技能、项目和实习经历。

【硬性要求】

1. 只能依据输入资料中的真实信息提出建议。
2. 不得虚构实习经历、项目经历、技术栈、业务指标、用户数量、性能提升或工作成果。
3. 如果资料缺少某项证据，应明确说明需要候选人补充或核实，不得擅自补全。
4. 不得把“建议学习某项技能”写成“候选人已经掌握该技能”。
5. 修改项目描述时，应保留原项目的真实范围和技术事实。
6. 如果原文没有量化指标，不得自行编造百分比、金额、访问量或效率提升。
7. 对不确定的信息使用“建议补充”“需要核实”等表述。
8. 建议应结合目标岗位，而不是泛泛地要求候选人全面提升自己。
9. 原则上优先利用已有经历调整表达顺序和重点，不要为了匹配岗位而捏造经历。
10. 输出中不能包含 Markdown 代码块、前置解释或 JSON 以外的内容。

【字段含义】

- target_position：目标岗位名称。
- overall_summary：整体评价，说明简历与目标岗位的主要关系。
- resume_positioning：简历定位建议，概括应该突出哪些真实优势。
- resume_summary：
  - current_assessment：当前简历摘要或定位的评价。
  - suggested_version：建议使用的个人简介或简历摘要文本。
  - evidence_basis：支撑该建议的已有事实。
- skills_recommendations：技能部分修改建议。每项建议尽量包含 skill、current_evidence、recommended_action、reason、priority 等字段。
- project_recommendations：项目经历调整建议。每项建议尽量包含 project_name、current_focus、recommended_focus、reason 等字段。
- internship_recommendations：实习经历调整建议。每项建议尽量包含 experience_name、current_focus、recommended_focus、reason 等字段。
- bullet_rewrites：简历条目改写草稿。每项建议尽量包含 section、original_text、suggested_text、reason 等字段。
- keywords_to_emphasize：应该突出且确实有事实依据的岗位关键词。
- do_not_claim：候选人不能未经核实就声称具备的能力或成果。
- priority_actions：候选人应该优先采取的修改行动。
- final_checklist：修改完成前的核查清单。

如果某个经历或技能不适用，可以返回空数组，不要为了填满字段而制造内容。

【输出长度限制】

11. 为保证 JSON 完整，严格控制输出规模：
    - skills_recommendations 最多 8 项；
    - project_recommendations 最多 5 项；
    - internship_recommendations 最多 3 项；
    - bullet_rewrites 最多 5 项；
    - keywords_to_emphasize 最多 15 项；
    - do_not_claim 最多 8 项；
    - priority_actions 最多 6 项；
    - final_checklist 最多 8 项。
12. 每条建议具体、简洁，避免同一内容重复出现在多个字段。
13. 每个字段都必须保留。没有内容时，使用空字符串、空数组或空对象中符合字段类型的形式。
14. 必须输出一个完整闭合的 JSON 对象，不得在输出中途结束。
15. JSON 字符串内部如需表示双引号，必须进行合法转义。
16. 除规定字段外，不要在最外层增加额外字段。

【JSON 输出结构】

请按以下结构返回。它是结构模板，不是需要照抄的示例内容：

{schema_example}

【候选人、目标岗位和匹配分析资料】

{input_json}

请根据以上真实资料生成简历修改建议。
只输出合法、完整的 JSON 对象。
""".strip()


# ============================================================
# 6. 调用大语言模型
# ============================================================

def _get_chat_completions_url() -> str:
    """兼容基础 URL 是否以 /v1 结尾的不同配置方式。"""
    if not LLM_BASE_URL:
        raise RuntimeError(
            "未配置 LLM_BASE_URL，请检查 backend/.env。"
        )

    if LLM_BASE_URL.endswith("/chat/completions"):
        return LLM_BASE_URL

    return f"{LLM_BASE_URL}/chat/completions"


def _extract_message_content(response_data: dict) -> tuple[str, str]:
    """从 OpenAI 兼容格式的响应中提取模型内容和结束原因。"""
    try:
        choice = response_data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        finish_reason = str(choice.get("finish_reason") or "unknown")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "岗位简历建议模型响应结构异常："
            f"{exc}\n模型响应内容：\n"
            + json.dumps(
                response_data,
                ensure_ascii=False,
                indent=2,
                default=str,
            )[:5000]
        ) from exc

    # 正常情况下 content 是字符串。兼容部分接口返回内容块列表。
    if isinstance(content, list):
        content_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                content_parts.append(item)
            elif isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str):
                    content_parts.append(text_part)

        content = "\n".join(content_parts)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "岗位简历建议模型返回内容为空。\n模型响应内容：\n"
            + json.dumps(
                response_data,
                ensure_ascii=False,
                indent=2,
                default=str,
            )[:5000]
        )

    return content, finish_reason


async def build_resume_advice(
    profile: Any,
    job: Any,
    skill_match: Any,
    experience_match: Any,
) -> dict:
    """
    US09：根据候选人画像、目标岗位和匹配结果生成简历修改建议。

    参数：
    - profile：候选人画像
    - job：结构化目标岗位
    - skill_match：技能匹配结果
    - experience_match：项目和实习经历匹配结果

    返回标准化的简历建议字典。
    """
    if not LLM_API_KEY:
        raise RuntimeError(
            "未配置 LLM_API_KEY，请检查 backend/.env。"
        )

    if not LLM_MODEL:
        raise RuntimeError(
            "未配置 LLM_MODEL，请检查 backend/.env。"
        )

    prompt = build_resume_advice_prompt(
        profile=profile,
        job=job,
        skill_match=skill_match,
        experience_match=experience_match,
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一名严谨的技术岗位简历顾问。"
                    "只能基于用户提供的真实资料生成建议，"
                    "禁止编造经历、技能、指标或成果。"
                    "必须严格输出完整合法的 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 8192,
        "response_format": {
            "type": "json_object",
        },
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    url = _get_chat_completions_url()

    timeout = httpx.Timeout(
        timeout=120.0,
        connect=20.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "简历修改建议请求模型超时。"
                "请稍后重试，或检查模型服务状态和网络连接。"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "简历修改建议请求失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc

    if response.is_error:
        # 返回错误正文以辅助排查，但不输出请求头或 API 密钥。
        error_body = response.text[:3000]

        raise RuntimeError(
            "简历修改建议模型 API 请求失败："
            f"HTTP {response.status_code}\n"
            f"响应内容：\n{error_body}"
        )

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "简历修改建议模型 API 返回的内容不是合法 JSON。\n"
            "响应正文（前 3000 字符）：\n"
            + response.text[:3000]
        ) from exc

    content, finish_reason = _extract_message_content(response_data)

    # 如果模型明确表示因达到输出长度上限而结束，就不要继续解析残缺内容。
    if finish_reason == "length":
        preview = content[:3000]

        if len(content) > 3000:
            preview += "\n...[剩余内容未显示]"

        raise RuntimeError(
            "模型输出可能因长度限制被截断，无法保证 JSON 完整。\n"
            "finish_reason=length\n"
            "请减少单次输出内容，或检查当前模型支持的输出长度。\n"
            "模型原始返回内容（前 3000 字符）：\n"
            f"{preview}"
        )

    # 解析模型内容；失败时将结束原因和原始内容放入异常，便于定位问题。
    try:
        result = extract_json(content)
    except ValueError as exc:
        preview = content[:5000]

        if len(content) > 5000:
            preview += "\n...[剩余内容未显示]"

        raise RuntimeError(
            f"{exc}\n"
            f"finish_reason={finish_reason}\n"
            "模型原始返回内容：\n"
            f"{preview}"
        ) from exc

    # 标准化字段，确保页面能够稳定读取预期结构。
    advice = normalize_resume_advice(
        result=result,
        job=job,
    )

    return advice
