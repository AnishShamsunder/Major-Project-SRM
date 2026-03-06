# Drug Interaction Checker — Cheat Sheet

---

## Abstract

Drug-drug interactions (DDIs) are a leading cause of adverse drug events, yet patients and caregivers rarely have fast, reliable access to interaction information at the point of decision. This project builds a **Retrieval-Augmented Generation (RAG)** system that combines dense vector search over a curated medical knowledge base with a large language model to answer natural-language DDI queries in real time.

The knowledge base is constructed from four complementary sources: the DrugBank DDI dataset (191,543 labelled interaction sentences), DrugBank compound vocabulary (17,432 drugs with synonyms), FDA regulatory data (products, applications, therapeutic equivalence), and 15 PubMed MEDLINE XML files (~2.5 GB of biomedical abstracts). All text is chunked, embedded using the BAAI/bge-small-en-v1.5 bi-encoder model (384-dimensional vectors), and stored in a Pinecone serverless vector index (207,083 vectors, cosine similarity).

At query time, both drug names are embedded and the top-k most semantically relevant chunks are retrieved from Pinecone. These chunks are passed as grounding context to Gemini 2.5 Flash Lite with a strict system prompt that prohibits the model from using any knowledge outside the retrieved context. The response is structured into three sections: interaction effects, physiological body impact, and safer alternatives. A React + Vite frontend with drug-name autocomplete (51,264 names) provides a user-facing interface.

The system is entirely context-grounded — it explicitly refuses to speculate or answer outside the knowledge base — making it suitable as a decision-support tool for informed medication review.

---

## Steps Followed

### Phase 1 — Project Setup
- Defined RAG architecture: embeddings → vector DB → LLM generation
- Created core Python modules: `embeddings.py`, `pinecone_client.py`, `ingest.py`, `retrieval.py`, `main.py`
- Set up FastAPI server with `/ingest`, `/query`, `/health` endpoints
- Configured Pinecone serverless index (`drug-rag-index`, 384-dim, cosine, AWS us-east-1)

### Phase 2 — Data Analysis & Parser Development
- Audited all files under `data/raw/` to understand formats and scale
- Wrote format-specific loaders for every source:
  - **CSV** — DDI sentences (`DrugA and DrugB interact: …`) and DrugBank vocabulary
  - **TSV** — FDA `.txt` files with per-file prose formatters (Products, Applications, TE, Marketing Status)
  - **SDF** — MDL V2000 state-machine parser extracting annotation blocks
  - **XML** — PubMed MEDLINE using `iterparse` streaming to handle 150–200 MB files without OOM
- Added `EXCLUDED_FILENAMES` set to skip 4 noisy FDA admin files (~52K unwanted vectors removed)

### Phase 3 — Prose Conversion
- Converted all structured rows into natural-language sentences before chunking
- Ensured every chunk reads as coherent English for better embedding quality
- Applied `RecursiveCharacterTextSplitter` (1800 chars, 200 overlap) across all sources

### Phase 4 — Ingestion Pipeline & Performance
- Initial approach (embed all chunks at once) estimated 6+ hours
- Rewrote to rolling batches: 512 chunks → embed (batch_size=256) → upsert → next batch
- Detected NVIDIA RTX 3050 6 GB GPU; installed PyTorch 2.6.0+cu124 for CUDA 12.4 acceleration
- Full ingestion completed: **207,083 vectors** upserted to Pinecone

### Phase 5 — Query Pipeline & Bug Fix
- Identified Pinecone 8.x SDK breaking change: responses are typed objects, not plain dicts
- Fixed `response.get("matches", [])` → `response.matches`, `match["score"]` → `match.score` etc.
- Confirmed end-to-end retrieval returning cosine similarity scores ~0.75–0.82 for DDI queries

### Phase 6 — Gemini Integration
- Added `gemini_client.py` using `google-genai` SDK (v1.66.0)
- Designed strict system prompt: context-only, no hallucination, explicit fallback messages
- Added `/ask` endpoint: retrieve top-k chunks → generate structured Gemini answer
- Output format: three sections — interaction effects / body impact / alternatives

### Phase 7 — System Prompt Refinement
- Tightened rules to reject out-of-scope questions
- Added **"What happens in the body"** section for physiological plain-language explanation
- Set temperature=0.2 for factual consistency; max_output_tokens=1024

### Phase 8 — React Frontend
- Built React 18 + Vite 6 frontend proxying `/api/*` → `http://localhost:8000`
- `DrugForm`: two drug inputs with live autocomplete against 51,264 drug names from DrugBank vocabulary
- `ResultCard`: three-section colour-coded display (red=interaction, amber=body, green=alternatives), collapsible source chunks with similarity scores
- Extracted `drugs.json` (51,264 names) from `drugbank vocabulary.csv` into `frontend/public/`

### Phase 9 — Finalisation
- Created `.gitignore` (excludes `data/raw/`, `venv/`, `node_modules/`, `.env`, logs)
- Wrote `README.md` with full setup instructions
- Pushed to GitHub: `https://github.com/AkashChintaluri/Major-Project-SRM`

---

## Endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/ask` | `{"query": "...", "top_k": 5}` | Full RAG — retrieve + Gemini answer |
| `POST` | `/query` | `{"query": "...", "top_k": 5}` | Raw vector search only |
| `POST` | `/ingest` | _(none)_ | Re-ingest all files in `data/` |
| `GET`  | `/health` | _(none)_ | Liveness check → `{"status":"ok"}` |

**`top_k` range:** 1–20  
**Quick test:**
```powershell
'{"query":"warfarin and aspirin","top_k":5}' | Out-File q.json -Encoding utf8 -NoNewline
curl.exe -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "@q.json"
```

---

## Configuration

### Backend — `main.py` / `.env`

| Variable | Default | Description |
|----------|---------|-------------|
| `PINECONE_API_KEY` | _(required)_ | Pinecone API key |
| `PINECONE_INDEX_NAME` | `drug-rag-index` | Pinecone index name |
| `GEMINI_API_KEY` | _(required)_ | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model ID |

### Embeddings — `embeddings.py`

| Setting | Value |
|---------|-------|
| Model | `BAAI/bge-small-en-v1.5` |
| Dimension | 384 |
| Device | `cuda` if GPU available, else `cpu` |
| Embed batch size | 256 (GPU) |
| Query prefix | `"Represent this sentence for searching relevant passages: "` |

### Chunking — `ingest.py`

| Setting | Value |
|---------|-------|
| `CHUNK_SIZE` | 1800 chars (~450 tokens) |
| `CHUNK_OVERLAP` | 200 chars (~50 tokens) |
| `INGEST_BATCH_SIZE` | 512 chunks per embed+upsert cycle |
| Upsert batch (Pinecone) | 100 vectors per API call |

### Pinecone Index — `pinecone_client.py`

| Setting | Value |
|---------|-------|
| Cloud | AWS `us-east-1` |
| Tier | Serverless (free) |
| Metric | Cosine similarity |
| Vectors | 207,083 |
| Storage used | ~0.83 GB / 2 GB limit |

### Gemini — `gemini_client.py`

| Setting | Value |
|---------|-------|
| Temperature | 0.2 |
| Max output tokens | 1024 |
| System constraints | Context-only, no outside knowledge |

---

## Knowledge Base Sources

| File | Format | Records | Notes |
|------|--------|---------|-------|
| `drug_drug_interactions.csv` | CSV | 191,543 rows | DDI sentences |
| `drugbank vocabulary.csv` | CSV | 17,432 rows | Drug identity + synonyms |
| `open structures.sdf` | MDL V2000 SDF | 12,316 molecules | Annotation blocks only |
| `fda/*.txt` | TSV | ~14K rows | Products, applications, TE |
| `pubmed/*.xml` | MEDLINE XML | 15 files ~2.5 GB | Title + abstract, streamed |

**Excluded (noisy/admin):**
- `SubmissionPropertyType.txt`
- `Join_Submission_ActionTypes_Lookup.txt`
- `ApplicationDocs.txt`
- `Submissions.txt`

---

## Response Format

Every `/ask` response has three sections:

```
**What happens when both are taken together:**
- Directional interaction effects from the knowledge base

**What happens in the body:**
- Physiological effects in plain language (organs, symptoms, measurable changes)

**What can be done instead:**
- Alternatives or safer options mentioned in the knowledge base
  (or explicit "not found" message if absent)

⚠️ Consult a licensed healthcare professional before making any medication changes.
```

---

## Running the Project

```powershell
# Backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm run dev
# → http://localhost:3000
```

---

## Key Demo Pairs

| Drug 1 | Drug 2 | Interaction |
|--------|--------|-------------|
| Warfarin | Aspirin | ↑ bleeding risk |
| Simvastatin | Amlodipine | ↑ statin levels, myopathy |
| Sertraline | Ibuprofen | ↑ GI bleeding |
| Clopidogrel | Omeprazole | ↓ antiplatelet efficacy |
| Metformin | Ciprofloxacin | Blood glucose effects |
| Warfarin | Rifampicin | ↓ warfarin serum concentration |

---

## File Map

```
main.py            FastAPI app, lifespan startup, endpoints
embeddings.py      Model load, embed_documents(), embed_query()
pinecone_client.py Pinecone init, get_or_create_index(), upsert, query
ingest.py          Parsers for CSV/TSV/SDF/XML, chunker, rolling batch upsert
retrieval.py       retrieve(query, top_k) → list of result dicts
gemini_client.py   generate_answer(query, chunks) → structured string
frontend/
  src/App.jsx              Root, fetch logic, error handling
  src/components/
    DrugForm.jsx/.css      Two inputs + autocomplete (51K drug names)
    ResultCard.jsx/.css    Three-section result display
  public/drugs.json        51,264 drug names for autocomplete
.env                       API keys (never committed)
.env.example               Template for new machines
requirements.txt           Python deps
```

---

## Dependencies

### Python
```
fastapi · uvicorn · sentence-transformers · pinecone
python-dotenv · langchain-text-splitters · google-genai · torch
```

### Node
```
react 18 · react-dom · vite 6 · @vitejs/plugin-react
```
