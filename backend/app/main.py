from pathlib import Path
import json
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.parsers import extract_text
from app.profile_service import build_profile


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
STATIC_DIR = BACKEND_DIR / "static"
DATA_DIR = BACKEND_DIR / "data"
PROFILE_FILE = DATA_DIR / "profile.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CareerPilot AI - US01",
    version="0.2.0",
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

MAX_FILE_SIZE = int(
    os.getenv("MAX_FILE_SIZE_MB", "10")
) * 1024 * 1024


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "careerpilot",
    }


@app.get("/api/profile")
def get_profile():
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
async def save_profile(profile: dict):
    if not isinstance(profile, dict):
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


@app.post("/api/resume/parse")
async def parse_resume(
    file: UploadFile = File(...),
):
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
        temp_path = Path(tmp.name)

    try:
        text = extract_text(temp_path)

        if len(text.strip()) < 30:
            raise HTTPException(
                status_code=422,
                detail=(
                    "未提取到足够文本，请确认简历不是纯扫描图片。"
                ),
            )

        profile = await build_profile(text)

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
        temp_path.unlink(
            missing_ok=True
        )