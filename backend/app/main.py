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
from app.match_service import match_profile_to_job
from app.skill_service import match_skills_to_profile
from app.experience_service import match_experiences_to_job
from app.conclusion_service import build_explainable_conclusion
from app.gap_service import build_gap_analysis


# ============================================================
# 基础配置
# ============================================================

load_dotenv()

app = FastAPI(
    title="CareerPilot AI",
    version="0.8.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "static"
DATA_DIR = BASE_DIR.parent / "data"
PROFILE_FILE = DATA_DIR / "profile.json"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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


class ConclusionRequest(BaseModel):
    job: dict


class GapAnalysisRequest(BaseModel):
    job: dict


# ============================================================
# 公共工具：读取个人画像
# ============================================================

def load_saved_profile() -> dict:
    """
    从 backend/data/profile.json 读取用户个人画像。
    """

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

    if not isinstance(profile, dict) or not profile:
        raise HTTPException(
            status_code=400,
            detail="当前个人画像为空或格式错误，请重新上传简历。",
        )

    return profile


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


@app.get("/conclusion")
def conclusion_page():
    return FileResponse(
        STATIC_DIR / "conclusion.html"
    )


@app.get("/gap")
def gap_page():
    return FileResponse(
        STATIC_DIR / "gap.html"
    )


# ============================================================
# 健康检查
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "0.8.0",
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
    file: UploadFile = File(...),
):
    """
    上传 PDF / DOCX 简历，提取文本并构建个人画像。
    解析成功后自动保存个人画像。
    """

    suffix = Path(
        file.filename or ""
    ).suffix.lower()

    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail="仅支持 PDF 或 DOCX。",
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="上传文件为空。",
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
        text = extract_text(path)

        if len(text.strip()) < 30:
            raise HTTPException(
                status_code=422,
                detail=(
                    "未提取到足够文本，"
                    "请确认文件不是纯扫描图片。"
                ),
            )

        profile = await build_profile(text)

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
            detail=f"简历解析失败：{str(e)}",
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
    data: JDParseRequest,
):
    """
    将原始 JD 转换为结构化岗位信息。
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
        job = await build_job(text)

        return {
            "text_length": len(text),
            "job": job,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"JD 解析失败：{str(e)}",
        )


# ============================================================
# US04：岗位硬性条件匹配
# ============================================================

@app.post("/api/jd/match")
async def match_job(
    data: MatchRequest,
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

    profile = load_saved_profile()

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
            detail=f"硬性条件匹配失败：{str(e)}",
        )


# ============================================================
# US05：技能匹配
# ============================================================

@app.post("/api/jd/skills-match")
async def match_skills(
    data: SkillMatchRequest,
):
    """
    将岗位技能要求与用户技能、项目经历、
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

    profile = load_saved_profile()

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
            detail=f"技能匹配失败：{str(e)}",
        )


# ============================================================
# US06：项目 / 实习经历匹配
# ============================================================

@app.post("/api/jd/experience-match")
async def match_experience(
    data: ExperienceMatchRequest,
):
    """
    将岗位工作内容与用户项目、实习经历进行匹配。
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

    responsibilities = job.get(
        "responsibilities",
        [],
    )

    if (
        not isinstance(responsibilities, list)
        or not responsibilities
    ):
        raise HTTPException(
            status_code=400,
            detail="岗位信息缺少工作内容，无法进行经历匹配。",
        )

    profile = load_saved_profile()

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
            detail=f"项目 / 实习经历匹配失败：{str(e)}",
        )


# ============================================================
# US07：可解释岗位匹配结论
# ============================================================

@app.post("/api/jd/conclusion")
async def get_job_conclusion(
    data: ConclusionRequest,
):
    """
    综合 US04、US05、US06 的分析结果，生成可解释结论。
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
            detail="岗位信息缺少岗位名称，请先完成 JD 解析。",
        )

    profile = load_saved_profile()

    try:
        hard_match = await match_profile_to_job(
            profile,
            job,
        )

        skill_match = await match_skills_to_profile(
            profile,
            job,
        )

        experience_match = await match_experiences_to_job(
            profile,
            job,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "生成综合结论时，"
                "前置匹配服务执行失败："
                f"{str(e)}"
            ),
        )

    try:
        conclusion = build_explainable_conclusion(
            job=job,
            hard_match=hard_match,
            skill_match=skill_match,
            experience_match=experience_match,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"综合结论生成失败：{str(e)}",
        )

    return {
        "profile": profile,
        "job": job,
        "hard_match": hard_match,
        "skill_match": skill_match,
        "experience_match": experience_match,
        "conclusion": conclusion,
    }


# ============================================================
# US08：能力 Gap 与补强建议
# ============================================================

@app.post("/api/jd/gap-analysis")
async def analyze_job_gaps(
    data: GapAnalysisRequest,
):
    """
    基于用户画像以及 US05、US06 的实际匹配结果，
    分析技能缺口、经历缺口和后续补强计划。

    US08 不直接把技能缺口等同于用户一定不会该技能，
    而是区分能力不足与现有资料缺少证据两种情况。
    """

    # --------------------------------------------------------
    # 1. 检查岗位信息
    # --------------------------------------------------------

    job = data.job

    if not isinstance(job, dict):
        raise HTTPException(
            status_code=400,
            detail="岗位信息格式错误。",
        )

    if not job.get("position"):
        raise HTTPException(
            status_code=400,
            detail="岗位信息缺少岗位名称，请先完成 JD 解析。",
        )

    # --------------------------------------------------------
    # 2. 读取用户个人画像
    # --------------------------------------------------------

    profile = load_saved_profile()

    # --------------------------------------------------------
    # 3. 重新执行 US05 技能匹配
    # --------------------------------------------------------

    try:
        skill_match = await match_skills_to_profile(
            profile,
            job,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "能力 Gap 分析前，"
                "US05 技能匹配失败："
                f"{str(e)}"
            ),
        )

    # --------------------------------------------------------
    # 4. 重新执行 US06 经历匹配
    # --------------------------------------------------------

    try:
        experience_match = await match_experiences_to_job(
            profile,
            job,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "能力 Gap 分析前，"
                "US06 经历匹配失败："
                f"{str(e)}"
            ),
        )

    # --------------------------------------------------------
    # 5. 生成 Gap 分析和补强计划
    # --------------------------------------------------------

    try:
        gap_analysis = await build_gap_analysis(
            profile=profile,
            job=job,
            skill_match=skill_match,
            experience_match=experience_match,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"能力 Gap 分析失败：{str(e)}",
        )

    # --------------------------------------------------------
    # 6. 返回数据
    # --------------------------------------------------------

    return {
        "profile": profile,
        "job": job,
        "skill_match": skill_match,
        "experience_match": experience_match,
        "gap_analysis": gap_analysis,
    }


# ============================================================
# 本地启动
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )