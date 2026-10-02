# CareerPilot AI - US01

Sprint 1 / US01：简历上传与解析。

上传 PDF/DOCX → 提取文本 → 调用 OpenAI-compatible LLM → 生成结构化个人求职画像 → 网页展示结果。

## 运行

cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

编辑 .env 后：
uvicorn app.main:app --reload

浏览器：http://127.0.0.1:8000
