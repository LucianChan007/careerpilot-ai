# ============================================================
# US08：能力 Gap 与补强建议服务
#
# 输入：
# 1. 用户个人画像
# 2. 目标岗位 JD
# 3. US05 技能匹配结果
# 4. US06 项目 / 实习经历匹配结果
#
# 输出：
# 1. 技能缺口
# 2. 经历缺口
# 3. 优先级计划
# 4. 可以优先开展的行动
# 5. 现有优势与风险
# ============================================================

import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

from app.gap_schema import GAP_ANALYSIS_SCHEMA


# ============================================================
# 环境变量
# ============================================================

load_dotenv()

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.deepseek.com",
)

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    "",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "deepseek-chat",
)


# ============================================================
# JSON 工具
# ============================================================

def extract_json(content: str) -> dict:
    """
    解析 LLM 返回的 JSON。

    兼容：
    1. 直接返回 JSON
    2. Markdown JSON 代码块
    3. JSON 前后存在少量解释文字
    """

    if not content or not content.strip():
        raise ValueError(
            "Gap 分析模型返回内容为空。"
        )

    content = content.strip()

    # --------------------------------------------------------
    # 1. 直接解析
    # --------------------------------------------------------

    try:
        result = json.loads(content)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # 2. 去除 Markdown 代码块
    # --------------------------------------------------------

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()

    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # 3. 尝试截取 JSON 对象
    # --------------------------------------------------------

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if (
        start >= 0
        and end > start
    ):
        try:
            result = json.loads(
                cleaned[start:end + 1]
            )

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Gap 分析模型返回内容无法解析为 JSON。"
    )


# ============================================================
# 标准化工具
# ============================================================

def as_list(value: Any) -> list:
    """将输入转换成列表。"""

    if isinstance(value, list):
        return value

    if value is None or value == "":
        return []

    return [value]


def to_text(value: Any) -> str:
    """把字段转换成适合展示的文本。"""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        return "；".join(
            to_text(item)
            for item in value
            if to_text(item)
        )

    if isinstance(value, dict):
        return "；".join(
            f"{key}：{to_text(item)}"
            for key, item in value.items()
            if item not in (None, "", [], {})
        )

    return str(value)


def normalize_gap_item(item: Any) -> dict:
    """
    标准化一个缺口条目。

    输出字段统一，方便前端展示。
    """

    if isinstance(item, str):
        return {
            "name": item,
            "priority": "P1",
            "severity": "中",
            "job_requirement": "",
            "current_evidence": "",
            "gap_description": item,
            "suggestion": "",
            "estimated_period": "",
            "acceptance_criteria": "",
        }

    if not isinstance(item, dict):
        return {
            "name": to_text(item) or "未命名缺口",
            "priority": "P1",
            "severity": "中",
            "job_requirement": "",
            "current_evidence": "",
            "gap_description": "",
            "suggestion": "",
            "estimated_period": "",
            "acceptance_criteria": "",
        }

    return {
        "name": to_text(
            item.get("name")
            or item.get("skill")
            or item.get("title")
            or item.get("requirement")
            or "未命名缺口"
        ),

        "priority": to_text(
            item.get("priority")
            or "P1"
        ),

        "severity": to_text(
            item.get("severity")
            or "中"
        ),

        "job_requirement": to_text(
            item.get("job_requirement")
            or item.get("requirement")
        ),

        "current_evidence": to_text(
            item.get("current_evidence")
            or item.get("evidence")
        ),

        "gap_description": to_text(
            item.get("gap_description")
            or item.get("description")
        ),

        "suggestion": to_text(
            item.get("suggestion")
            or item.get("recommendation")
        ),

        "estimated_period": to_text(
            item.get("estimated_period")
            or item.get("timeframe")
        ),

        "acceptance_criteria": to_text(
            item.get("acceptance_criteria")
            or item.get("deliverable")
        ),
    }


def normalize_priority_item(item: Any) -> dict:
    """标准化阶段计划中的一项任务。"""

    if isinstance(item, str):
        return {
            "priority": "P1",
            "title": item,
            "reason": "",
            "actions": [],
            "deliverable": "",
            "estimated_period": "",
        }

    if not isinstance(item, dict):
        return {
            "priority": "P1",
            "title": to_text(item) or "待补强任务",
            "reason": "",
            "actions": [],
            "deliverable": "",
            "estimated_period": "",
        }

    actions = as_list(
        item.get("actions", [])
    )

    return {
        "priority": to_text(
            item.get("priority") or "P1"
        ),

        "title": to_text(
            item.get("title")
            or item.get("name")
            or "待补强任务"
        ),

        "reason": to_text(
            item.get("reason")
        ),

        "actions": [
            to_text(action)
            for action in actions
            if to_text(action)
        ],

        "deliverable": to_text(
            item.get("deliverable")
            or item.get("acceptance_criteria")
        ),

        "estimated_period": to_text(
            item.get("estimated_period")
            or item.get("timeframe")
        ),
    }


def normalize_gap_analysis(result: dict) -> dict:
    """
    将模型输出转换成固定数据结构。

    不以默认文字伪装成已经完成的分析。
    """

    result = result or {}

    schema = GAP_ANALYSIS_SCHEMA

    skill_gaps = [
        normalize_gap_item(item)
        for item in as_list(
            result.get("skill_gaps")
        )
    ]

    experience_gaps = [
        normalize_gap_item(item)
        for item in as_list(
            result.get("experience_gaps")
        )
    ]

    priority_plan = [
        normalize_priority_item(item)
        for item in as_list(
            result.get("priority_plan")
        )
    ]

    quick_wins = [
        to_text(item)
        for item in as_list(
            result.get("quick_wins")
        )
        if to_text(item)
    ]

    strengths = [
        to_text(item)
        for item in as_list(
            result.get("strengths_to_leverage")
        )
        if to_text(item)
    ]

    risk_notes = [
        to_text(item)
        for item in as_list(
            result.get("risk_notes")
        )
        if to_text(item)
    ]

    gap_level = to_text(
        result.get("overall_gap_level")
    )

    valid_levels = {
        "高",
        "中",
        "低",
        "信息不足",
    }

    if gap_level not in valid_levels:
        gap_level = "信息不足"

    normalized = {
        **schema,

        "overall_gap_level": gap_level,

        "overall_summary": to_text(
            result.get("overall_summary")
        ),

        "skill_gaps": skill_gaps,

        "experience_gaps": experience_gaps,

        "priority_plan": priority_plan,

        "quick_wins": quick_wins,

        "strengths_to_leverage": strengths,

        "risk_notes": risk_notes,

        "summary": to_text(
            result.get("summary")
        ),
    }

    return normalized


# ============================================================
# Prompt 构建
# ============================================================

def build_gap_prompt(
    profile: dict,
    job: dict,
    skill_match: dict,
    experience_match: dict,
) -> str:
    """
    生成 US08 分析 Prompt。

    重点是识别差距，并把差距转换为可执行计划。
    """

    output_schema = json.dumps(
        GAP_ANALYSIS_SCHEMA,
        ensure_ascii=False,
        indent=2,
    )

    input_data = {
        "profile": profile,
        "job": job,
        "skill_match": skill_match,
        "experience_match": experience_match,
    }

    input_text = json.dumps(
        input_data,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是 CareerPilot AI 的能力差距分析助手。

你的任务是：
根据目标岗位要求、用户当前个人画像、US05 技能匹配结果、
US06 项目 / 实习经历匹配结果，识别用户目前的能力差距，
并给出有优先级、可执行、可验收的补强计划。

输出必须是 JSON 对象，结构如下：

{output_schema}

字段解释：

一、overall_gap_level

整体 Gap 程度，仅允许：
- 高
- 中
- 低
- 信息不足

该字段表示当前岗位要求与用户已有证据之间的差距程度，
不是对用户能力或个人价值的评价。

二、overall_summary

概述用户目前最主要的能力差距，以及判断依据。

三、skill_gaps

记录技能缺口。每个条目必须尽量包含：

- name：缺少或需要加强的技能
- priority：P0 / P1 / P2
- severity：高 / 中 / 低
- job_requirement：岗位对应要求
- current_evidence：用户现有技能或相关经历证据
- gap_description：当前能力与岗位要求之间的差距
- suggestion：可执行的学习或练习建议
- estimated_period：合理的计划时间范围
- acceptance_criteria：如何判断这项补强工作已经完成

技能确实匹配且证据充分的项目，不要列为技能缺口。
对于“信息不足”的技能，要区分“用户确实不会”和“当前资料没有提供证据”。

四、experience_gaps

记录项目或实习经历方面的缺口。

如果用户具备某项技能，但个人画像没有明确体现实际使用该技能完成相关项目，
应说明“缺乏直接项目证据”，而不能擅自认定用户不会该技能。

建议可以是未来项目，但必须明确它是待完成的计划，
不能写成用户已经完成的经历。

五、priority_plan

将主要补强工作整理成有先后顺序的任务列表。

每个条目包括：
- priority：P0 / P1 / P2
- title：任务名称
- reason：为什么要做
- actions：建议具体步骤，使用字符串数组
- deliverable：预期可检查的产出
- estimated_period：建议时间范围

优先级定义：
- P0：对岗位核心要求或关键缺口影响很大的事项
- P1：重要的次级技能或项目证据补强
- P2：优化项或加分项

不要为了凑够任务数量强行生成无关计划。
如果现有资料不足，应明确说明原因。

六、quick_wins

列出可以较快完成且具有实际价值的行动。
例如：整理真实项目中的技术栈、补充可核实的功能实现说明、
使用针对性练习验证某项技能。

不要建议用户把没有做过的工作写进简历。

七、strengths_to_leverage

总结可以用于补强其他能力的现有优势。
必须能在用户画像或匹配结果中找到依据。

八、risk_notes

列出需要进一步确认的事项，例如：
- JD 信息不完整
- 项目技术栈不明确
- 简历中缺少实现细节
- 岗位要求超出目前已有证据

九、summary

用清晰的中文总结 Gap 判断和行动方向。

严格遵守以下原则：

1. 用户的当前能力只能由现有个人画像和分析结果支持。
2. 不得虚构用户已经掌握的技能、做过的项目或实习经历。
3. 必须区分“能力缺失”和“证据不足”。
4. 不能把仅仅出现在岗位 JD 中的技能当作用户已具备的技能。
5. 不要把通用建议写得过于空泛，例如只写“提升编程能力”。
6. 建议应尽可能对应岗位实际要求。
7. 不得编造招聘方未提供的硬性要求。
8. 薪资不属于技能差距，不需要纳入 Gap 分析。
9. 时间范围仅为计划建议，不是完成效果的保证。
10. 只输出 JSON，不输出 Markdown 代码块或额外解释。

以下是实际输入数据：

{input_text}
"""


# ============================================================
# 调用 DeepSeek / OpenAI 兼容接口
# ============================================================

async def build_gap_analysis(
    profile: dict,
    job: dict,
    skill_match: dict,
    experience_match: dict,
) -> dict:
    """
    构建能力 Gap 分析结果。

    此函数本身不重新执行 US05 或 US06。
    调用方应传入已有的实际匹配分析结果。
    """

    if not LLM_API_KEY:
        raise RuntimeError(
            "未配置 LLM_API_KEY，请检查 backend/.env。"
        )

    if not isinstance(profile, dict) or not profile:
        raise ValueError(
            "个人画像为空，无法分析能力 Gap。"
        )

    if not isinstance(job, dict) or not job.get("position"):
        raise ValueError(
            "岗位信息不完整，缺少岗位名称。"
        )

    if not isinstance(skill_match, dict):
        raise ValueError(
            "US05 技能匹配结果格式错误。"
        )

    if not isinstance(experience_match, dict):
        raise ValueError(
            "US06 经历匹配结果格式错误。"
        )

    prompt = build_gap_prompt(
        profile=profile,
        job=job,
        skill_match=skill_match,
        experience_match=experience_match,
    )

    base_url = LLM_BASE_URL.rstrip("/")

    if base_url.endswith("/chat/completions"):
        url = base_url
    else:
        url = base_url + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的能力差距分析助手。"
                    "必须根据输入证据进行分析，"
                    "不能虚构已有经历，且必须严格输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_object"
        },
    }

    # --------------------------------------------------------
    # 1. 调用模型
    # --------------------------------------------------------

    async with httpx.AsyncClient(
        timeout=90.0
    ) as client:

        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )

    # --------------------------------------------------------
    # 2. 检查 HTTP 状态
    # --------------------------------------------------------

    if response.status_code != 200:
        raise RuntimeError(
            "Gap 分析模型调用失败："
            f"HTTP {response.status_code}；"
            f"返回内容：{response.text}"
        )

    # --------------------------------------------------------
    # 3. 解析 API JSON
    # --------------------------------------------------------

    try:
        response_data = response.json()

    except Exception as e:
        raise RuntimeError(
            "Gap 分析模型返回的 API 内容不是有效 JSON："
            f"{e}；返回内容：{response.text}"
        )

    # --------------------------------------------------------
    # 4. 获取模型消息
    # --------------------------------------------------------

    try:
        content = (
            response_data["choices"][0]
            ["message"]["content"]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as e:
        raise RuntimeError(
            "Gap 分析模型响应缺少 message.content："
            f"{e}；响应内容："
            + json.dumps(
                response_data,
                ensure_ascii=False,
            )
        )

    # --------------------------------------------------------
    # 5. 提取和标准化结果
    # --------------------------------------------------------

    result = extract_json(
        content
    )

    normalized = normalize_gap_analysis(
        result
    )

    # 不允许悄悄返回完全空白的分析结果。
    if (
        not normalized["overall_summary"]
        and not normalized["summary"]
        and not normalized["skill_gaps"]
        and not normalized["experience_gaps"]
        and not normalized["priority_plan"]
    ):
        raise RuntimeError(
            "Gap 分析返回结果缺少有效内容。"
            "请检查模型原始响应是否完整。"
        )

    return normalized