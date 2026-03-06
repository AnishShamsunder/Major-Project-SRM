# Drug Interaction Checker — RAG System

A Retrieval-Augmented Generation (RAG) system for drug-drug interaction (DDI) lookup. Enter two medications and get a structured answer — what happens when they are taken together, what the body experiences, and what alternatives exist — grounded entirely in a curated medical knowledge base.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector DB | Pinecone (serverless, 207K vectors) |
| LLM | Gemini 2.5 Flash Lite |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + Vite |

---

## Knowledge Base

Ingested from four data sources (~207,083 vectors):

| Source | Description |
|---|---|
| DrugBank DDI CSV | 191,543 drug-drug interaction sentences |
| DrugBank Vocabulary | 17,432 drug compound records |
| FDA Regulatory Data | Products, applications, marketing status, therapeutic equivalence |
| PubMed Abstracts | 15 MEDLINE XML files (~2.5 GB), title + abstract indexed |

---

## Project Structure

```
├── main.py              # FastAPI server (endpoints: /ingest, /query, /ask, /health)
├── embeddings.py        # BAAI/bge-small-en-v1.5, auto GPU/CPU detection
├── pinecone_client.py   # Pinecone connection, upsert, query
├── ingest.py            # Data loaders for CSV, TSV, SDF, XML formats
├── retrieval.py         # Query pipeline: embed → vector search → parse
├── gemini_client.py     # Gemini 2.5 Flash Lite, system prompt, answer generation
├── requirements.txt
├── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── components/
    │       ├── DrugForm.jsx   # Two drug inputs with autocomplete (51K drug names)
    │       └── ResultCard.jsx # Three-section structured result display
    └── public/
        └── drugs.json         # 51,264 drug names for autocomplete
```

---

## API Endpoints

### `POST /ask` — Full RAG pipeline
```json
{
  "query": "What happens if I take warfarin and aspirin together?",
  "top_k": 5
}
```
Returns a structured answer with three sections:
- **What happens when both are taken together** — directional interaction effects
- **What happens in the body** — physiological effects in plain language
- **What can be done instead** — alternatives mentioned in the knowledge base

### `POST /query` — Raw vector search
Returns the top-k most similar chunks from Pinecone without LLM generation.

### `GET /health`
Returns `{"status": "ok"}`.

---

## Setup & Running

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Clone
```bash
git clone https://github.com/AkashChintaluri/Major-Project-SRM
cd Major-Project-SRM
```

### 2. Python environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Optional — NVIDIA GPU acceleration:**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Environment variables
```powershell
copy .env.example .env
```

Edit `.env`:
```
PINECONE_API_KEY=<your Pinecone API key>
PINECONE_INDEX_NAME=drug-rag-index
GEMINI_API_KEY=<your Google AI Studio API key>
GEMINI_MODEL=gemini-2.5-flash-lite
```

> The Pinecone index is cloud-hosted — no re-ingestion needed on a new machine.

### 4. Frontend
```powershell
cd frontend
npm install
```

### 5. Start

**Terminal 1 — Backend:**
```powershell
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```powershell
cd frontend
npm run dev
```

Open **http://localhost:3000**

---

## Example Interactions to Try

| Drug 1 | Drug 2 | Expected finding |
|---|---|---|
| Warfarin | Aspirin | Increased bleeding risk |
| Simvastatin | Amlodipine | Elevated statin levels, myopathy risk |
| Sertraline | Ibuprofen | Increased GI bleeding risk |
| Clopidogrel | Omeprazole | Reduced antiplatelet efficacy |
| Metformin | Ciprofloxacin | Blood glucose effects |

---

## Notes

- The system answers **only** from retrieved context. It will not hallucinate or use outside knowledge.
- If the knowledge base lacks information for a combination, it says so explicitly.
- All responses end with a disclaimer to consult a healthcare professional.
