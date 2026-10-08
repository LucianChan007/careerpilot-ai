import json
import os
import re

import httpx
from dotenv import load_dotenv

from app.jd_schema import JOB_SCHEMA


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
# Prompt
# ============================================================

def build_jd_prompt(
    jd_text: str,
) -> str:

    schema_text = json.dumps(
        JOB_SCHEMA,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是一个专业的大学生求职岗位分析助手。

你的任务是：
将用户提供的原始岗位 JD 解析成严格的结构化 JSON。

必须严格按照下面的 JSON 结构输出：

{schema_text}

字段含义如下：

1. company
公司名称。
如果 JD 中没有明确出现公司名称，填写空字符串。

2. position
岗位名称。
这是非常重要的字段。
必须从 JD 标题、岗位名称、职位名称等位置提取。
例如：
"C++开发实习生"
"Java后端开发实习生"
"大数据开发工程师"
不能因为 JD 中信息较多而留空。

3. salary
薪资信息。
例如：
"100-200元/天"
"8-12K/月"
薪资只用于展示，不属于硬性匹配条件。

4. responsibilities
岗位职责 / 工作内容。
必须重点识别：
"岗位职责"
"工作职责"
"职位描述"
"工作内容"
"你将负责"
"主要工作"
等部分。

每一条职责单独作为数组中的一个字符串。

5. required_education
学历要求。
例如：
"本科"
"硕士"
"学历不限"

6. required_major
专业要求。
必须是数组。
例如：
["计算机科学与技术", "软件工程"]

如果没有明确专业要求，返回 []。

7. required_skills
岗位要求的技能。
必须从任职要求、技能要求、技术栈等内容中提取。

例如：
["C/C++", "Qt", "Python", "Git"]

8. experience
经验要求。
例如：
["桌面开发经验"]
["后端开发经验"]
["有实习经验优先"]

9. location
工作地点。
例如：
"广州"
"深圳"
"上海"

10. internship_days
实习时间要求。
例如：
"5天/周3个月"
"每周至少4天"

11. other_requirements
其他要求。
例如：
["良好的沟通能力", "责任心强"]

重要规则：

第一，必须尽可能提取 JD 原文中的信息，不要自行编造。

第二，岗位职责和任职要求必须区分：
"岗位职责"属于 responsibilities。
"任职要求"中的技术能力属于 required_skills。
"任职要求"中的工作经验属于 experience。

第三：
薪资必须放到 salary。
不能放到 other_requirements。

第四：
如果岗位名称出现在 JD 第一行，例如：

C++开发实习生
100-200元/天
广州

那么 position 必须是：
"C++开发实习生"

第五：
如果存在：

岗位职责:
1. 使用C/C++语言进行软件开发。
2. 与团队合作进行软件测试。
3. 参与软件整体架构设计。

必须解析成：

"responsibilities": [
    "使用C/C++语言进行软件开发。",
    "与团队合作进行软件测试。",
    "参与软件整体架构设计。"
]

第六：
只输出 JSON。
禁止输出解释文字。
禁止输出 Markdown。
禁止输出 ```json。

原始 JD：

--------------------
{jd_text}
--------------------
"""


# ============================================================
# 提取 JSON
# ============================================================

def extract_json(
    content: str,
) -> dict:

    if not content:
        raise ValueError(
            "LLM 返回内容为空。"
        )

    content = content.strip()

    # --------------------------------------------------------
    # 1. 直接解析
    # --------------------------------------------------------

    try:
        result = json.loads(
            content
        )

        if isinstance(
            result,
            dict,
        ):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # 2. 去除 Markdown JSON 代码块
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
        result = json.loads(
            cleaned
        )

        if isinstance(
            result,
            dict,
        ):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # 3. 从文本中寻找第一个 JSON 对象
    # --------------------------------------------------------

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        json_text = cleaned[
            start:end + 1
        ]

        try:
            result = json.loads(
                json_text
            )

            if isinstance(
                result,
                dict,
            ):
                return result

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "LLM 返回内容无法解析为 JSON。"
    )


# ============================================================
# 标准化岗位结构
# ============================================================

def normalize_job(
    job: dict,
) -> dict:

    result = {
        "company": "",
        "position": "",
        "salary": "",
        "responsibilities": [],
        "required_education": "",
        "required_major": [],
        "required_skills": [],
        "experience": [],
        "location": "",
        "internship_days": "",
        "other_requirements": [],
    }

    # --------------------------------------------------------
    # 复制已有字段
    # --------------------------------------------------------

    for key in result.keys():

        if key in job:
            result[key] = job[key]

    # --------------------------------------------------------
    # 标准化字符串字段
    # --------------------------------------------------------

    string_fields = [
        "company",
        "position",
        "salary",
        "required_education",
        "location",
        "internship_days",
    ]

    for key in string_fields:

        value = result[key]

        if value is None:
            result[key] = ""

        elif not isinstance(
            value,
            str,
        ):
            result[key] = str(
                value
            )


    # --------------------------------------------------------
    # 标准化数组字段
    # --------------------------------------------------------

    list_fields = [
        "responsibilities",
        "required_major",
        "required_skills",
        "experience",
        "other_requirements",
    ]

    for key in list_fields:

        value = result[key]

        if value is None:
            result[key] = []

        elif isinstance(
            value,
            str,
        ):

            if value.strip():
                result[key] = [
                    value.strip()
                ]
            else:
                result[key] = []

        elif not isinstance(
            value,
            list,
        ):

            result[key] = [
                str(value)
            ]


    # --------------------------------------------------------
    # 清理数组中的空字符串
    # --------------------------------------------------------

    for key in list_fields:

        cleaned = []

        for item in result[key]:

            if isinstance(
                item,
                str,
            ):

                item = item.strip()

                if item:
                    cleaned.append(
                        item
                    )

            else:

                cleaned.append(
                    item
                )

        result[key] = cleaned


    return result


# ============================================================
# 调用 LLM
# ============================================================

async def build_job(
    jd_text: str,
) -> dict:

    if not LLM_API_KEY:
        raise RuntimeError(
            "未配置 LLM_API_KEY，请检查 backend/.env"
        )

    if not jd_text.strip():
        raise ValueError(
            "JD 内容不能为空。"
        )

    prompt = build_jd_prompt(
        jd_text
    )

    headers = {
        "Authorization":
            f"Bearer {LLM_API_KEY}",

        "Content-Type":
            "application/json",
    }

    payload = {
        "model": LLM_MODEL,

        "messages": [
            {
                "role": "system",
                "content":
                    "你是一个严谨的岗位 JD 结构化解析助手。",
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
    # API 地址
    # --------------------------------------------------------

    base_url = LLM_BASE_URL.rstrip("/")

    if not base_url.endswith(
        "/chat/completions"
    ):
        url = (
            base_url
            + "/chat/completions"
        )
    else:
        url = base_url

    # --------------------------------------------------------
    # 调用 DeepSeek
    # --------------------------------------------------------

    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:

        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )

    # --------------------------------------------------------
    # HTTP 错误
    # --------------------------------------------------------

    if response.status_code != 200:

        raise RuntimeError(
            "LLM API 调用失败："
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    # --------------------------------------------------------
    # 解析 API JSON
    # --------------------------------------------------------

    try:

        response_data = response.json()

    except Exception as e:

        raise RuntimeError(
            "LLM API 返回内容不是有效 JSON："
            f"{e}\n"
            f"{response.text}"
        )

    # --------------------------------------------------------
    # 获取模型文本
    # --------------------------------------------------------

    try:

        content = (
            response_data
            ["choices"]
            [0]
            ["message"]
            ["content"]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as e:

        raise RuntimeError(
            "LLM 返回结构异常："
            f"{e}\n"
            f"{json.dumps(response_data, ensure_ascii=False, indent=2)}"
        )

    # --------------------------------------------------------
    # JSON 解析
    # --------------------------------------------------------

    job = extract_json(
        content
    )

    # --------------------------------------------------------
    # 标准化
    # --------------------------------------------------------

    job = normalize_job(
        job
    )

    # --------------------------------------------------------
    # 关键字段校验
    #
    # 不再允许“解析成功但 position 为空”
    # --------------------------------------------------------

    if not job["position"]:

        raise RuntimeError(
            "JD 解析结果缺少岗位名称 position。"
            "\n模型原始返回：\n"
            + content
        )

    if not job["responsibilities"]:

        raise RuntimeError(
            "JD 解析结果缺少岗位职责 responsibilities。"
            "\n模型原始返回：\n"
            + content
        )

    return job