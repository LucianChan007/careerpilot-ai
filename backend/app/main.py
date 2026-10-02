from pathlib import Path
import os, tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.parsers import extract_text
from app.profile_service import build_profile

load_dotenv()
app = FastAPI(title="CareerPilot AI - US01", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/resume/parse")
async def parse_resume(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(400, "仅支持 PDF 或 DOCX。")
    data = await file.read()
    if not data:
        raise HTTPException(400, "文件为空。")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, f"文件超过 {MAX_FILE_SIZE // 1024 // 1024} MB。")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = Path(tmp.name)
    try:
        text = extract_text(path)
        if len(text.strip()) < 30:
            raise HTTPException(422, "未提取到足够文本，请确认不是纯扫描图片。")
        profile = await build_profile(text)
        return {"filename": file.filename, "text_length": len(text), "profile": profile}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        path.unlink(missing_ok=True)
