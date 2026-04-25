# databricks_news_ai

Demo chatbot đọc báo theo kiến trúc hybrid:

- Databricks phụ trách pipeline dữ liệu báo chí
- Ứng dụng local phụ trách indexing, retrieval và hỏi đáp RAG

Project này không dùng Databricks Vector Search và cũng không dùng Databricks Model Serving. Vector store được lưu local bằng ChromaDB, còn mô hình sinh câu trả lời chạy qua Ollama.

## Mục tiêu

Hệ thống gồm 2 phần tách biệt:

1. Pipeline dữ liệu trên Databricks
   - Crawl bài viết từ các trang báo
   - Parse HTML
   - Chuẩn hóa URL
   - Làm sạch nội dung
   - Deduplicate
   - Ghi ra bảng Delta sạch để downstream sử dụng

2. Ứng dụng RAG local
   - Đọc dữ liệu từ bảng `main.news_ai.articles_clean`
   - Chunk nội dung bài viết
   - Tạo embedding local
   - Lưu vector vào ChromaDB local
   - Retrieve các chunk liên quan
   - Gọi Ollama để sinh câu trả lời kèm nguồn

## Kiến trúc tổng quan

Luồng dữ liệu:

```text
News websites
-> Databricks jobs
-> main.news_ai.news_raw_documents
-> main.news_ai.news_articles
-> main.news_ai.articles_clean
-> local embedding
-> ChromaDB local
-> RAG chatbot
```

Ba job chính trên Databricks:

- `crawl_news_job`: crawl bài viết và ghi dữ liệu raw vào `main.news_ai.news_raw_documents`
- `parse_and_canonicalize_job`: parse HTML, normalize URL, clean text, dedup và ghi ra `main.news_ai.news_articles`
- `build_articles_clean_job`: build lại bảng output cuối `main.news_ai.articles_clean`

Nguồn crawl hiện tại:

- `vnexpress`
- `cafef`
- `genk`
- `diendandoanhnghiep`

## Bảng đầu ra chính

Ứng dụng local đọc dữ liệu từ:

`main.news_ai.articles_clean`

Schema tối thiểu:

```sql
article_id STRING
source STRING
url STRING
canonical_url STRING
title STRING
summary_raw STRING
content STRING
category STRING
published_at TIMESTAMP
crawled_at TIMESTAMP
content_hash STRING
dedup_group_id STRING
is_duplicate BOOLEAN
created_at TIMESTAMP
updated_at TIMESTAMP
```

## Cấu trúc repo

```text
databricks_news_ai/
|-- app/
|   |-- jobs/                 # Python jobs cho Databricks
|   |-- ingestion/            # crawler, parser, service crawl
|   |-- processing/           # clean, canonicalize, dedup
|   |-- databricks/           # session, Delta table, repository
|   |-- local_ai/             # local RAG, Chroma, Ollama, Streamlit
|   |-- domain/               # model/schema dùng chung
|   |-- repositories/
|   `-- utils/
|-- notebooks/jobs/           # notebook wrapper cho Databricks Workflow
|-- data/chroma/              # vector store local của ChromaDB
|-- tests/
|-- databricks_files/         # bản mirror để bundle/deploy lên Databricks
|-- databricks.yml
|-- requirements.txt
|-- requirements-local.txt
`-- .env.example
```

## Thành phần chính trong local AI

- `app/local_ai/demo_chatbot.py`: CLI để index dữ liệu và hỏi đáp
- `app/local_ai/streamlit_app.py`: frontend Streamlit
- `app/local_ai/pipeline.py`: lắp các thành phần embedding, vector store, RAG service
- `app/local_ai/vector_store.py`: thao tác với ChromaDB
- `app/local_ai/embeddings.py`: tạo embedding bằng `sentence-transformers/all-MiniLM-L6-v2`
- `app/local_ai/rag_service.py`: luồng retrieve, rerank, build context và sinh câu trả lời
- `app/local_ai/ollama_client.py`: gọi Ollama qua HTTP
- `app/local_ai/prompt_builder.py`: dựng prompt từ context truy hồi
- `app/local_ai/reranker.py`: rerank hybrid theo vector score và keyword overlap

## Mô hình đang dùng

- LLM trả lời: `mistral:7b`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: `ChromaDB`

## Điều kiện trước khi chạy

Cần có:

- Databricks workspace
- SQL Warehouse trên Databricks
- Databricks CLI
- Python 3.11+
- Ollama cài trên máy local
- Windows PowerShell

## Cài đặt môi trường local

### 1. Tạo file `.env`

```powershell
Copy-Item .env.example .env
```

Điền các biến bắt buộc:

```dotenv
DATABRICKS_SERVER_HOSTNAME=<workspace-hostname>
DATABRICKS_HTTP_PATH=<sql-warehouse-http-path>
DATABRICKS_TOKEN=<personal-access-token>
DATABRICKS_ARTICLES_TABLE=main.news_ai.articles_clean

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

Không commit file `.env`.

### 2. Tạo virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu dùng `cmd`:

```cmd
.venv\Scripts\activate.bat
```

### 3. Cài dependency

Dependency crawl cơ bản:

```powershell
python -m pip install -r requirements.txt
```

Dependency local chatbot:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

## Chạy pipeline trên Databricks

### Đăng nhập Databricks

```powershell
databricks auth login --host https://dbc-c8b4d0cc-1da2.cloud.databricks.com
```

### Validate và deploy bundle

```powershell
databricks bundle validate
databricks bundle deploy
```

### Chạy 3 job theo thứ tự

```powershell
databricks bundle run crawl_news_job
databricks bundle run parse_and_canonicalize_job
databricks bundle run build_articles_clean_job
```

Sau khi chạy xong, dữ liệu sạch nằm ở:

`main.news_ai.articles_clean`

## Tham số crawler chính

`crawl_news_job` hiện dùng mode `category_pagination`:

```text
crawl_mode=category_pagination
discover_categories=true
max_pages_per_category=6
stop_after_empty_pages=3
stop_after_duplicate_pages=3
request_delay_seconds=1
```

Trong mode này:

- crawler tự discover category từ source
- duyệt qua từng page của mỗi category
- lấy article links
- bỏ qua link đã tồn tại trong Delta
- crawl trang chi tiết
- lưu raw payload và metadata

`MAX_ARTICLES_PER_SOURCE` hiện là tham số legacy và không còn đóng vai trò giới hạn số bài trong mode này.

## Chạy crawler local

Dry-run:

```powershell
python -m app.jobs.crawl_news_job --source_names vnexpress --dry_run true --max_pages_per_category 6
```

Crawl local và ghi Delta:

```powershell
python -m app.jobs.crawl_news_job --source_names vnexpress --dry_run false --max_pages_per_category 6 --stop_after_empty_pages 3 --stop_after_duplicate_pages 3 --request_delay_seconds 1
```

## Cài và chạy Ollama

Pull model:

```powershell
ollama pull mistral:7b
```

Chạy server:

```powershell
ollama serve
```

Nếu Ollama không chạy, hệ thống sẽ fallback sang extractive answer. Tuy nhiên, để chất lượng trả lời tốt hơn, nên chạy đầy đủ `mistral:7b`.

## Chạy chatbot local bằng CLI

### Build vector index

```powershell
python -m app.local_ai.demo_chatbot --index --limit 100
```

Lệnh này sẽ:

- đọc dữ liệu từ `main.news_ai.articles_clean`
- bỏ qua bài duplicate hoặc content rỗng
- chunk bài viết
- tạo embedding bằng `sentence-transformers/all-MiniLM-L6-v2`
- lưu vector vào ChromaDB local tại `data/chroma`

### Build lại index từ đầu

```powershell
python -m app.local_ai.demo_chatbot --reset_index --index --limit 100
```

### Hỏi đáp

```powershell
python -m app.local_ai.demo_chatbot --question "Tin AI mới nhất có gì đáng chú ý?"
```

### Build index và hỏi đáp trong cùng một lệnh

```powershell
python -m app.local_ai.demo_chatbot --index --limit 100 --question "Tóm tắt tin công nghệ mới nhất"
```

### Một số ví dụ khác

```powershell
python -m app.local_ai.demo_chatbot --question "Tóm tắt bài có tiêu đề \"...\""
python -m app.local_ai.demo_chatbot --summarize_url "https://example.com/article"
python -m app.local_ai.demo_chatbot --question "Tin kinh doanh gần đây có gì nổi bật?" --top_k 5
python -m app.local_ai.demo_chatbot --index --limit 200 --chunk_size 800 --chunk_overlap 150
```

### Debug retrieval và prompt

```powershell
python -m app.local_ai.demo_chatbot --question "Tin AI mới nhất có gì đáng chú ý?" --debug_retrieval
python -m app.local_ai.demo_chatbot --question "Tin công nghệ gần đây nổi bật là gì?" --debug_prompt
```

## Chạy frontend Streamlit

Sau khi đã điền `.env` và cài dependency:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/local_ai/streamlit_app.py
```

UI hiện có:

- kiểm tra biến môi trường Databricks và Ollama
- build index từ `main.news_ai.articles_clean`
- reset Chroma collection
- hỏi đáp qua RAG
- hiển thị nguồn gồm:
  - `title`
  - `source`
  - `url`
  - `category`
  - `published_at`
  - `chunk_id`

## Biến môi trường quan trọng

- `DATABRICKS_SERVER_HOSTNAME`
- `DATABRICKS_HTTP_PATH`
- `DATABRICKS_TOKEN`
- `DATABRICKS_ARTICLES_TABLE`
- `LOCAL_EMBEDDING_MODEL`
- `CHROMA_PERSIST_DIR`
- `CHROMA_COLLECTION_NAME`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `RAG_TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

Giá trị mặc định trong `.env.example`:

```dotenv
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=news_articles
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
RAG_TOP_K=4
CHUNK_SIZE=700
CHUNK_OVERLAP=120
```

## Cách hệ thống trả lời câu hỏi

Luồng RAG local hiện tại:

1. Embed câu hỏi bằng model embedding local
2. Search vector trong ChromaDB
3. Rerank kết quả bằng vector score + keyword overlap + metadata overlap
4. Chọn các chunk đa dạng theo article
5. Build prompt từ context truy hồi
6. Gọi Ollama để sinh câu trả lời
7. Trả về answer kèm citations

Chống ảo giác hiện tại chủ yếu dựa vào:

- chỉ trả lời dựa trên context đã truy hồi
- lọc ứng viên theo ngưỡng score
- fallback về thông báo không đủ dữ liệu nếu context yếu
- fallback sang extractive answer nếu Ollama lỗi

## Quy trình chạy nhanh

Nếu muốn chạy nhanh từ đầu đến cuối:

1. Chạy 3 job Databricks
2. Điền `.env`
3. Cài dependency local
4. Chạy `ollama pull mistral:7b`
5. Chạy `ollama serve`
6. Build index:

```powershell
python -m app.local_ai.demo_chatbot --index --limit 100
```

7. Hỏi đáp:

```powershell
python -m app.local_ai.demo_chatbot --question "Tin công nghệ mới nhất có gì nổi bật?"
```

Hoặc mở Streamlit:

```powershell
python -m streamlit run app/local_ai/streamlit_app.py
```

## Giới hạn hiện tại

- Hệ thống đang ở mức demo RAG, chưa phải production system
- Chưa có API gateway/backend riêng, frontend Streamlit gọi logic Python trực tiếp
- Intent detection còn đơn giản và chủ yếu dựa vào keyword
- Reranker hiện là heuristic, chưa dùng learned reranker hoặc cross-encoder
- Chưa mạnh với câu hỏi mơ hồ, nhiều điều kiện hoặc cần suy luận nhiều bước
- ChromaDB đang lưu local, chưa tối ưu cho scale lớn hoặc nhiều người dùng
- Chất lượng phụ thuộc mạnh vào dữ liệu crawl và độ sạch của `articles_clean`

## Troubleshooting

### Lỗi `ModuleNotFoundError: No module named 'app'` trên Databricks

Nguyên nhân thường là notebook không thấy project root hoặc bundle cũ chưa deploy lại.

Chạy lại:

```powershell
databricks bundle deploy -t dev
databricks bundle run crawl_news_job -t dev
```

Nếu upload notebook thủ công trên Databricks Web, cần upload cả thư mục `app/` cùng với `notebooks/`.

### Local chatbot báo index rỗng

Nguyên nhân thường gặp:

- chưa chạy `build_articles_clean_job`
- `.env` sai `DATABRICKS_HTTP_PATH` hoặc token
- bảng `main.news_ai.articles_clean` không có dữ liệu
- chưa chạy `--index`

Kiểm tra lại:

```powershell
python -m app.local_ai.demo_chatbot --reset_index --index --limit 100
```

### Local chatbot không gọi được Ollama

Đảm bảo đã chạy:

```powershell
ollama pull mistral:7b
ollama serve
```

Và `.env` có:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

### Streamlit mở được nhưng hỏi đáp lỗi

Nguyên nhân thường là:

- Chroma index chưa được build
- Ollama chưa chạy
- Databricks credentials chưa đúng

Hãy build index trước trong UI hoặc bằng CLI.

## Ghi chú kỹ thuật

- Local AI app chỉ đọc từ `articles_clean`, không ghi ngược lại Databricks
- ChromaDB chỉ lưu local trong `data/chroma`
- `chunk_id` ổn định theo dạng `article_id:chunk_index`, nên upsert sẽ ghi đè đúng chunk thay vì phải xóa toàn bộ index
- Nếu Ollama lỗi, app sẽ fallback sang extractive answer và vẫn hiển thị nguồn
