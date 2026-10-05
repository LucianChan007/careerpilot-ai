from pathlib import Path
import json
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.parsers import extract_text
from app.profile_service import build_profile
from app.jd_service import build_job


# =========================
# 环境变量
# =========================

load_dotenv()


# =========================
# 路径配置
# =========================

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent

STATIC_DIR = BACKEND_DIR / "static"
DATA_DIR = BACKEND_DIR / "data"

PROFILE_FILE = DATA_DIR / "profile.json"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================
# FastAPI
# =========================

app = FastAPI(
    title="CareerPilot AI",
    version="0.3.0",
    description="CareerPilot AI：大学生实习求职研究与投递助理",
)


# =========================
# 静态文件
# =========================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# =========================
# 配置
# =========================

MAX_FILE_SIZE = (
    int(
        os.getenv(
            "MAX_FILE_SIZE_MB",
            "10",
        )
    )
    * 1024
    * 1024
)


# =========================
# Pydantic 请求模型
# =========================

class JDRequest(BaseModel):
    jd_text: str


# =========================
# 首页
# =========================

@app.get("/")
def index():
    return FileResponse(
        STATIC_DIR / "index.html"
    )

@app.get("/jd")
def jd_page():
    return FileResponse(
        STATIC_DIR / "jd.html"
    )

# =========================
# 健康检查
# =========================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "careerpilot",
    }


# ============================================================
# US02：个人画像
# ============================================================

@app.get("/api/profile")
def get_profile():
    """
    读取保存的个人画像。
    """

    if not PROFILE_FILE.exists():
        return {}

    try:
        with PROFILE_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="个人画像文件格式损坏。",
        ) from exc


@app.put("/api/profile")
async def save_profile(
    profile: dict,
):
    """
    保存个人画像。
    """

    if not isinstance(
        profile,
        dict,
    ):
        raise HTTPException(
            status_code=400,
            detail="个人画像必须是 JSON 对象。",
        )

    try:
        with PROFILE_FILE.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                profile,
                f,
                ensure_ascii=False,
                indent=2,
            )

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="个人画像保存失败。",
        ) from exc

    return {
        "success": True,
        "message": "个人画像保存成功。",
        "profile": profile,
    }


# ============================================================
# US01：简历上传与 AI 解析
# ============================================================

@app.post("/api/resume/parse")
async def parse_resume(
    file: UploadFile = File(...),
):
    """
    上传 PDF / DOCX 简历，
    提取文本并调用 LLM 生成结构化个人画像。
    """

    # -------------------------
    # 1. 文件格式检查
    # -------------------------

    suffix = Path(
        file.filename or ""
    ).suffix.lower()

    if suffix not in {
        ".pdf",
        ".docx",
    }:
        raise HTTPException(
            status_code=400,
            detail="仅支持 PDF 或 DOCX。",
        )

    # -------------------------
    # 2. 读取文件
    # -------------------------

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="文件为空。",
        )

    # -------------------------
    # 3. 文件大小检查
    # -------------------------

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"文件超过 "
                f"{MAX_FILE_SIZE // 1024 // 1024} MB。"
            ),
        )

    # -------------------------
    # 4. 创建临时文件
    # -------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:
        tmp.write(data)
        temp_path = Path(
            tmp.name
        )

    try:
        # -------------------------
        # 5. 提取简历文本
        # -------------------------

        text = extract_text(
            temp_path
        )

        if len(
            text.strip()
        ) < 30:
            raise HTTPException(
                status_code=422,
                detail=(
                    "未提取到足够文本，"
                    "请确认简历不是纯扫描图片。"
                ),
            )

        # -------------------------
        # 6. AI 解析个人画像
        # -------------------------

        profile = await build_profile(
            text
        )

        # -------------------------
        # 7. 返回结果
        # -------------------------

        return {
            "filename": file.filename,
            "text_length": len(text),
            "profile": profile,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        # -------------------------
        # 8. 删除临时文件
        # -------------------------

        temp_path.unlink(
            missing_ok=True
        )


# ============================================================
# US03：JD 输入与结构化解析
# ============================================================

@app.post("/api/jd/parse")
async def parse_jd(
    data: JDRequest,
):
    """
    接收岗位 JD 原文，
    调用 LLM 生成结构化岗位信息。
    """

    # -------------------------
    # 1. 获取 JD 文本
    # -------------------------

    jd_text = data.jd_text.strip()

    # -------------------------
    # 2. 空内容检查
    # -------------------------

    if not jd_text:
        raise HTTPException(
            status_code=400,
            detail="JD 内容不能为空。",
        )

    # -------------------------
    # 3. 最小长度检查
    # -------------------------

    if len(jd_text) < 30:
        raise HTTPException(
            status_code=400,
            detail=(
                "JD 内容过短，"
                "请输入完整岗位描述。"
            ),
        )

    # -------------------------
    # 4. 最大长度检查
    # -------------------------

    if len(jd_text) > 20000:
        raise HTTPException(
            status_code=413,
            detail=(
                "JD 内容过长，"
                "暂时限制为 20000 个字符。"
            ),
        )

    try:
        # -------------------------
        # 5. AI 结构化解析
        # -------------------------

        job = await build_job(
            jd_text
        )

        # -------------------------
        # 6. 返回结构化岗位
        # -------------------------

        return {
            "text_length": len(jd_text),
            "job": job,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc