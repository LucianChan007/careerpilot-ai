from pathlib import Path
import os
import json
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.parsers import extract_text
from app.profile_service import build_profile
from app.jd_service import parse_jd
from app.match_service import match_profile_to_job
from app.skill_service import match_skills_to_profile
from app.experience_service import match_experiences_to_job


# ============================================================
# 基础配置
# ============================================================

load_dotenv()

app = FastAPI(
    title="CareerPilot AI",
    version="0.6.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "static"
DATA_DIR = BASE_DIR.parent / "data"
PROFILE_FILE = DATA_DIR / "profile.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

MAX_FILE_SIZE = (
    int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    * 1024
    * 1024
)


# ============================================================
# 请求模型
# ============================================================

class JDParseRequest(BaseModel):
    text: str


class MatchRequest(BaseModel):
    job: dict


class SkillMatchRequest(BaseModel):
    job: dict


class ExperienceMatchRequest(BaseModel):
    job: dict


# ============================================================
# 页面路由
# ============================================================

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


@app.get("/match")
def match_page():
    return FileResponse(
        STATIC_DIR / "match.html"
    )


@app.get("/skills")
def skills_page():
    return FileResponse(
        STATIC_DIR / "skills.html"
    )


@app.get("/experience")
def experience_page():
    return FileResponse(
        STATIC_DIR / "experience.html"
    )


# ============================================================
# 健康检查
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# 个人画像
# ============================================================

@app.get("/api/profile")
def get_profile():
    """
    获取当前保存的用户个人画像。
    """

    if not PROFILE_FILE.exists():
        return {
            "profile": {}
        }

    try:
        profile = json.loads(
            PROFILE_FILE.read_text(
                encoding="utf-8"
            )
        )

        return {
            "profile": profile
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取个人画像失败：{str(e)}",
        )


@app.put("/api/profile")
async def update_profile(data: dict):
    """
    更新并保存用户个人画像。
    """

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail="个人画像格式错误。",
        )

    try:
        PROFILE_FILE.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "message": "个人画像保存成功。",
            "profile": data,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"保存个人画像失败：{str(e)}",
        )


# ============================================================
# US01：简历解析
# ============================================================

@app.post("/api/resume/parse")
async def parse_resume(
    file: UploadFile = File(...)
):
    """
    上传 PDF / DOCX 简历，
    提取文本并调用 LLM 生成结构化个人画像。

    解析完成后自动保存到：
    backend/data/profile.json
    """

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

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="文件为空。",
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"文件超过 "
                f"{MAX_FILE_SIZE // 1024 // 1024} MB。"
            ),
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:

        tmp.write(data)
        path = Path(tmp.name)

    try:
        # ----------------------------------------------------
        # 1. 提取简历文本
        # ----------------------------------------------------

        text = extract_text(path)

        if len(text.strip()) < 30:
            raise HTTPException(
                status_code=422,
                detail=(
                    "未提取到足够文本，"
                    "请确认文件不是纯扫描图片。"
                ),
            )

        # ----------------------------------------------------
        # 2. LLM 生成结构化画像
        # ----------------------------------------------------

        profile = await build_profile(text)

        # ----------------------------------------------------
        # 3. 自动保存个人画像
        # ----------------------------------------------------

        PROFILE_FILE.write_text(
            json.dumps(
                profile,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "filename": file.filename,
            "text_length": len(text),
            "profile": profile,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:
        path.unlink(
            missing_ok=True
        )


# ============================================================
# US03：JD 结构化解析
# ============================================================

@app.post("/api/jd/parse")
async def parse_job(
    data: JDParseRequest
):
    """
    将原始岗位描述解析为结构化岗位信息。
    """

    text = data.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="岗位描述不能为空。",
        )

    if len(text) < 10:
        raise HTTPException(
            status_code=400,
            detail="岗位描述内容过短。",
        )

    try:
        job = await parse_jd(text)

        return {
            "text_length": len(text),
            "job": job,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# US04：岗位硬性条件匹配
# ============================================================

@app.post("/api/jd/match")
async def match_job(
    data: MatchRequest
):
    """
    将岗位硬性条件与用户个人画像进行匹配。
    """

    job = data.job

    if not isinstance(job, dict):
        raise HTTPException(
            status_code=400,
            detail="岗位信息格式错误。",
        )

    if not job.get("position"):
        raise HTTPException(
            status_code=400,
            detail="岗位信息缺少岗位名称。",
        )

    if not PROFILE_FILE.exists():
        raise HTTPException(
            status_code=400,
            detail="当前没有保存的个人画像，请先上传简历。",
        )

    try:
        profile = json.loads(
            PROFILE_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取个人画像失败：{str(e)}",
        )

    if not profile:
        raise HTTPException(
            status_code=400,
            detail="当前个人画像为空，请先上传简历。",
        )

    try:
        result = await match_profile_to_job(
            profile,
            job,
        )

        return {
            "profile": profile,
            "job": job,
            "match": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# US05：技能与岗位匹配
# ============================================================

@app.post("/api/jd/skills-match")
async def match_skills(
    data: SkillMatchRequest,
):
    """
    将岗位要求技能与用户技能、项目经历、
    实习经历进行逐项匹配。
    """

    job = data.job

    if not isinstance(job, dict):
        raise HTTPException(
            status_code=400,
            detail="岗位信息格式错误。",
        )

    if not job.get("position"):
        raise HTTPException(
            status_code=400,
            detail="岗位信息缺少岗位名称。",
        )

    if not PROFILE_FILE.exists():
        raise HTTPException(
            status_code=400,
            detail="当前没有保存的个人画像，请先上传简历。",
        )

    try:
        profile = json.loads(
            PROFILE_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取个人画像失败：{str(e)}",
        )

    if not profile:
        raise HTTPException(
            status_code=400,
            detail="当前个人画像为空，请先上传简历。",
        )

    try:
        result = await match_skills_to_profile(
            profile,
            job,
        )

        return {
            "profile": profile,
            "job": job,
            "skill_match": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# US06：项目 / 经历与岗位工作内容匹配
# ============================================================

@app.post("/api/jd/experience-match")
async def match_experience(
    data: ExperienceMatchRequest,
):
    """
    将岗位工作内容与用户项目经历、实习经历进行匹配。
    """

    # --------------------------------------------------------
    # 1. 获取岗位
    # --------------------------------------------------------

    job = data.job

    if not isinstance(
        job,
        dict,
    ):
        raise HTTPException(
            status_code=400,
            detail="岗位信息格式错误。",
        )

    # --------------------------------------------------------
    # 2. 检查岗位名称
    # --------------------------------------------------------

    if not job.get("position"):
        raise HTTPException(
            status_code=400,
            detail="岗位信息缺少岗位名称。",
        )

    # --------------------------------------------------------
    # 3. 检查岗位工作内容
    # --------------------------------------------------------

    responsibilities = job.get(
        "responsibilities",
        [],
    )

    if not isinstance(
        responsibilities,
        list,
    ) or not responsibilities:

        raise HTTPException(
            status_code=400,
            detail="岗位信息缺少工作内容，无法进行经历匹配。",
        )

    # --------------------------------------------------------
    # 4. 检查个人画像
    # --------------------------------------------------------

    if not PROFILE_FILE.exists():
        raise HTTPException(
            status_code=400,
            detail="当前没有保存的个人画像，请先上传简历。",
        )

    try:
        profile = json.loads(
            PROFILE_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取个人画像失败：{str(e)}",
        )

    # --------------------------------------------------------
    # 5. 检查个人画像是否为空
    # --------------------------------------------------------

    if not profile:
        raise HTTPException(
            status_code=400,
            detail="当前个人画像为空，请先上传简历。",
        )

    # --------------------------------------------------------
    # 6. 调用 US06 经历匹配服务
    # --------------------------------------------------------

    try:
        result = await match_experiences_to_job(
            profile,
            job,
        )

        return {
            "profile": profile,
            "job": job,
            "experience_match": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# 启动配置
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
