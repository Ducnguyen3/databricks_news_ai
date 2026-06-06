# Databricks News AI/RAG

Hệ thống hỏi đáp tin tức tiếng Việt sử dụng kiến trúc Lakehouse trên Databricks kết hợp Retrieval-Augmented Generation (RAG). Dữ liệu tin tức được crawl từ các báo điện tử, xử lý qua các lớp Bronze/Silver/Gold trên Delta Lake, đồng bộ sang ChromaDB local để truy hồi, sau đó được phục vụ qua FastAPI và giao diện React.

## Tính Năng Chính

- Hỏi đáp tin tức tiếng Việt theo ngôn ngữ tự nhiên.
- Truy hồi bài báo bằng hybrid search: vector search, BM25, metadata filtering và reranking.
- Route câu hỏi theo chủ đề, thực thể, mã cổ phiếu, thời gian và nhu cầu xem ảnh.
- Trả lời có trích dẫn nguồn, danh sách bài liên quan và ảnh bài báo nếu có.
- Hỗ trợ truy vấn ảnh như `cho tôi 4 ảnh về Ukraine` hoặc `ảnh chiến sự Trung Đông`.
- Debug trace cho pipeline RAG: query plan, retrieval, topic guard, rerank, prompt và generation.
- Frontend React hiển thị câu trả lời, nguồn, ảnh, bài liên quan và mở link bài báo gốc.

## Ví Dụ Câu Hỏi

```text
tin AI mới nhất
HPG có gì mới
tổng hợp tin chứng khoán tuần này
tình hình thế giới hôm nay
cho tôi 6 ảnh về Ukraine
cho tôi các ảnh liên quan đến tình hình chiến sự Trung Đông
```

## Kiến Trúc

```text
News websites
  -> Crawlers
  -> Bronze: main.news_ai.news_raw_documents
  -> Parse / clean / deduplicate / topic / entity / image extraction
  -> Silver: main.news_ai.news_articles + main.news_ai.news_article_images
  -> Gold: main.news_ai.articles_clean
  -> Chunking + embeddings
  -> ChromaDB local
  -> Query router + hybrid retrieval + metadata filter + topic guard + reranker
  -> Prompt builder / extractive fallback / Ollama
  -> FastAPI backend
  -> React frontend
```

Databricks Delta Lake là nguồn dữ liệu chính. ChromaDB trong `data/chroma` chỉ là vector index local và có thể rebuild từ bảng Gold.

## Cấu Trúc Thư Mục

| Path | Vai trò |
|---|---|
| `app/main.py` | FastAPI app, gồm `/health` và `/api/chat` |
| `app/config.py` | Load cấu hình từ `.env` |
| `app/ingestion` | Crawler, parser và service thu thập tin |
| `app/processing` | Entity extraction, taxonomy, image extraction |
| `app/databricks` | Delta schemas, Spark session, repositories |
| `app/jobs` | Job crawl, parse, canonicalize và rebuild Chroma |
| `app/local_ai` | Chunking, embedding, Chroma, query router, retriever, reranker, RAG service |
| `frontend` | React + TypeScript + Vite chat UI |
| `scripts` | Script kiểm tra/sửa Chroma |
| `tests` | Unit tests cho crawler, metadata, retrieval, RAG, frontend |
| `docs` | Runbook, báo cáo và tài liệu audit |

## Nguồn Dữ Liệu

Crawler hiện hỗ trợ:

- VNExpress
- CafeF
- GenK
- Diễn đàn Doanh nghiệp

Các bảng chính:

| Layer | Table | Mục đích |
|---|---|---|
| Bronze | `main.news_ai.news_raw_documents` | HTML/payload thô từ crawler |
| Silver | `main.news_ai.news_articles` | Bài viết đã parse và enrich |
| Images | `main.news_ai.news_article_images` | Metadata ảnh bài báo |
| Gold | `main.news_ai.articles_clean` | Bài viết sạch dùng cho RAG |

## Yêu Cầu Môi Trường

- Python 3.11+
- Node.js + npm
- Databricks SQL Warehouse và token hợp lệ
- Ollama local nếu muốn dùng LLM local
- Windows PowerShell hoặc terminal tương đương

Tạo môi trường Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu `requirements-local.txt` có dependency bổ sung:

```powershell
python -m pip install -r requirements-local.txt
```

Cài frontend:

```powershell
cd frontend
npm.cmd install
cd ..
```

## Cấu Hình `.env`

Tạo file cấu hình local:

```powershell
Copy-Item .env.example .env
```

Các biến quan trọng:

```dotenv
DATABRICKS_SERVER_HOSTNAME=<databricks-host>
DATABRICKS_HTTP_PATH=<sql-warehouse-http-path>
DATABRICKS_TOKEN=<databricks-token>
DATABRICKS_ARTICLES_TABLE=main.news_ai.articles_clean

LOCAL_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=news_articles

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_TIMEOUT_SECONDS=180

RAG_BROAD_RETRIEVE_TOP_N=30
RAG_TOP_K=4
RAG_MIN_SCORE=0.35
RAG_DEBUG=false
PROMPT_MAX_CONTEXT_CHARS=12000
```

Không commit `.env` vì file này chứa token và thông tin môi trường.

## Chạy Ollama

Kiểm tra Ollama:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

Pull model nếu chưa có:

```powershell
ollama pull qwen3:8b
```

Nếu dùng model khác, sửa `OLLAMA_MODEL` trong `.env` rồi restart backend.

## Chạy Backend

Từ root repo:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Gọi API chat:

```powershell
$body = @{
  message = "HPG có gì mới"
  top_k = 4
  debug = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/chat `
  -Method Post `
  -Body $body `
  -ContentType "application/json"
```

Response có cấu trúc chính:

```json
{
  "answer": "...",
  "intent": "entity_news",
  "topic": "economy_finance_stock",
  "query_plan": {},
  "sources": [],
  "images": [],
  "generated_image_prompts": [],
  "related_articles": [],
  "debug": {}
}
```

## Chạy Frontend

```powershell
cd frontend
npm.cmd run dev
```

Mặc định Vite chạy tại:

```text
http://localhost:5173
```

Frontend gọi backend qua API base URL mặc định:

```text
http://localhost:8000
```

Nếu backend chạy port khác, cấu hình biến môi trường frontend tương ứng trước khi start Vite.

## Đồng Bộ Gold Sang Chroma

Full rebuild:

```powershell
python -m app.local_ai.index_sync --rebuild_mode full
```

Incremental sync:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental
```

Một số lệnh hữu ích:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental --limit 100
python -m app.local_ai.index_sync --rebuild_mode incremental --source cafef
python -m app.local_ai.index_sync --rebuild_mode incremental --topic economy_finance_stock
python -m app.local_ai.index_sync --rebuild_mode full --chroma_path data/chroma
```

Kiểm tra Chroma:

```powershell
python scripts/inspect_chroma_health.py --chroma-dir data/chroma --collection news_articles
```

Smoke test RAG:

```powershell
python -m app.local_ai.rag_smoke_test --chroma_path data/chroma --collection_name news_articles
```

## Crawl Và Databricks Jobs

Dry run crawler local:

```powershell
python -m app.jobs.crawl_news_job --source_names vnexpress --dry_run true --max_pages_per_category 5
```

Crawl và ghi Delta:

```powershell
python -m app.jobs.crawl_news_job --source_names vnexpress --dry_run false --max_pages_per_category 5
```

Databricks bundle:

```powershell
databricks bundle validate
databricks bundle deploy
databricks bundle run crawl_news_job
databricks bundle run parse_and_canonicalize_job
```

## Query Router Debug

Kiểm tra route của một câu hỏi:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "from app.local_ai.query_router import route_query; import json; print(json.dumps(route_query('cho tôi 6 ảnh về Ukraine'), ensure_ascii=False, indent=2))"
```

Kết quả mong đợi:

```text
intent = media_lookup
primary_topic = world_geopolitics
need_images = true
image_limit = 6
```

## Test

Python unit tests:

```powershell
python -m unittest discover -s tests
```

Các test nhanh cho RAG/media/router:

```powershell
python -m unittest tests.test_query_router tests.test_media_retriever tests.test_rag_media_lookup tests.test_rag_media_response
```

Frontend:

```powershell
cd frontend
npm.cmd run build
npm.cmd run lint
```

## Demo Nên Chuẩn Bị

Khi báo cáo, nên demo ngắn các luồng sau:

1. Hỏi `HPG có gì mới` để thấy answer có nguồn.
2. Click một source để mở bài báo gốc.
3. Hỏi `cho tôi 4 ảnh về Ukraine` để thấy ảnh thật từ metadata/crawl.
4. Bật debug để xem `query_plan`, `retrieval`, `rerank`, `prompt`, `generation`.
5. Hỏi một câu ngoài dữ liệu để thấy hệ thống fallback thay vì bịa.

## Troubleshooting

### Port 8000 đang bị chiếm

Kiểm tra process:

```powershell
netstat -ano | findstr :8000
```

Tắt process theo PID:

```powershell
taskkill /PID <PID> /F
```

Trong PowerShell, thay `<PID>` bằng số thật, ví dụ:

```powershell
taskkill /PID 12345 /F
```

### Backend trả 200 nhưng Ollama lỗi

RAG service có fallback extractive answer nên `/api/chat` vẫn có thể trả 200. Kiểm tra Ollama:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

Nếu model không tồn tại:

```powershell
ollama pull qwen3:8b
```

### Backend dùng sai Chroma path

Kiểm tra:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Chạy lại backend với env đúng:

```powershell
$env:CHROMA_PERSIST_DIR="data/chroma"
$env:CHROMA_COLLECTION_NAME="news_articles"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Không có ảnh trong câu trả lời

Kiểm tra các điểm sau:

- Query có route `intent = media_lookup` và `need_images = true` không.
- Chroma metadata có `images_json`, `image_url` hoặc `has_images` không.
- Bảng `main.news_ai.news_article_images` có ảnh cho `article_id` tương ứng không.
- Nếu có bài liên quan nhưng thiếu ảnh, backend sẽ trả message phân biệt rõ thay vì nói chung chung là không đủ dữ liệu.

### Chroma index mismatch

Nếu health/audit báo lệch giữa SQLite embeddings và HNSW IDs, nên rebuild:

```powershell
python -m app.local_ai.index_sync --rebuild_mode full --chroma_path data/chroma
```

## Tài Liệu Liên Quan

- `docs/end_to_end_runbook.md`
- `docs/chroma_index_sync.md`
- `docs/hybrid_search.md`
- `docs/rag_smoke_test.md`
- `docs/rag_quality_eval.md`
- `docs/chunk_quality_audit.md`
- `docs/rag_pipeline_audit_prompt.md`
- `docs/report.md`

