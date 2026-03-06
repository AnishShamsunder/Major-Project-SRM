"""
ingest.py
---------
Data loading, text chunking, embedding generation, and Pinecone upsert.

Supported file types detected in data/raw/:
    drug_drug_interactions.csv  – Drug-drug interaction CSV (Drug 1, Drug 2, Description)
    drugbank vocabulary.csv     – DrugBank drug identity/synonym CSV
    open structures.sdf         – MDL V2000 SDF; only annotation blocks extracted
    fda/*.txt                   – Tab-delimited FDA files (treated as TSV)
    pubmed/*.xml                – MEDLINE PubmedArticleSet XML; title + abstract only
                                  (streamed with iterparse to handle 150-200 MB files)
    *.json                      – Generic JSON (list of strings or dicts)

Unknown extensions are skipped with a warning.
"""

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import embed_documents
from pinecone_client import upsert_vectors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

# Approximate token counts: 1 token ≈ 4 chars for English text.
# 400-500 tokens → ~1600-2000 chars; target midpoint (450 tokens → 1800 chars).
# 50-token overlap → ~200 chars
CHUNK_SIZE = 1800    # characters
CHUNK_OVERLAP = 200  # characters

# ---------------------------------------------------------------------------
# Files to exclude from ingestion
# These are pure regulatory admin / lookup tables with no clinical value.
# Including them would add ~52,000 noisy vectors that could pollute DDI queries.
# ---------------------------------------------------------------------------
EXCLUDED_FILENAMES: set[str] = {
    # 302,357 rows of submission property codes ("Null" values, no clinical text)
    "SubmissionPropertyType.txt",
    # 204,234 rows of submission-to-action-type join table (pure admin)
    "Join_Submission_ActionTypes_Lookup.txt",
    # 78,433 rows of document URLs and filing dates (no drug interaction content)
    "ApplicationDocs.txt",
    # 189,691 rows of approval status dates (regulatory timeline, not clinical)
    "Submissions.txt",
}

# ---------------------------------------------------------------------------
# Text splitter (LangChain RecursiveCharacterTextSplitter)
# ---------------------------------------------------------------------------
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def _load_drug_interactions_csv(path: Path) -> str:
    """
    Load drug_drug_interactions.csv.

    Columns: Drug 1, Drug 2, Interaction Description
    Each row is rendered as a human-readable sentence so that semantic search
    can match natural-language queries about drug interactions.

    Example output line:
        "Warfarin and Aspirin interact: Warfarin may increase the anticoagulant
         activities of Aspirin."
    """
    records: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            drug1 = row.get("Drug 1", "").strip()
            drug2 = row.get("Drug 2", "").strip()
            desc  = row.get("Interaction Description", "").strip()
            if drug1 and drug2 and desc:
                records.append(f"{drug1} and {drug2} interact: {desc}")
    return "\n".join(records)


def _load_drugbank_vocabulary_csv(path: Path) -> str:
    """
    Load drugbank vocabulary.csv.

    Columns: DrugBank ID, Accession Numbers, Common name, CAS, UNII,
             Synonyms, Standard InChI Key
    Each row is rendered as a short drug profile sentence.

    Example output:
        "Lepirudin (DrugBank ID: DB00001, CAS: 138068-37-8).
         Synonyms: [Leu1, Thr2]-63-desulfohirudin; Hirudin variant-1."
    """
    records: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            db_id   = row.get("DrugBank ID", "").strip()
            name    = row.get("Common name", "").strip()
            cas     = row.get("CAS", "").strip()
            synonyms = row.get("Synonyms", "").strip()

            line = f"{name} (DrugBank ID: {db_id}"
            if cas:
                line += f", CAS: {cas}"
            line += ")."
            if synonyms:
                line += f" Synonyms: {synonyms}."
            records.append(line)
    return "\n".join(records)


def _load_generic_csv(path: Path) -> str:
    """
    Fallback CSV loader for files not specifically recognised.
    Formats each row as 'ColumnName: value | ColumnName: value …' using the
    header row as labels, so context is preserved without column names being lost.
    """
    records: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            # No header – fall back to raw join
            fh.seek(0)
            plain = csv.reader(fh)
            return "\n".join(" | ".join(r) for r in plain)
        for row in reader:
            parts = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
            records.append(" | ".join(parts))
    return "\n".join(records)


def _load_csv(path: Path) -> str:
    """
    Route CSV files to the appropriate specialised loader based on filename,
    then fall back to the generic loader.
    """
    name_lower = path.name.lower()
    if "drug_drug_interactions" in name_lower:
        return _load_drug_interactions_csv(path)
    if "drugbank vocabulary" in name_lower or "drugbank_vocabulary" in name_lower:
        return _load_drugbank_vocabulary_csv(path)
    return _load_generic_csv(path)


def _is_tsv(path: Path) -> bool:
    """Return True if the first line of the file contains tab characters."""
    try:
        first_line = path.open(encoding="utf-8", errors="replace").readline()
        return "\t" in first_line
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Per-file prose formatters for FDA tab-delimited files
# ---------------------------------------------------------------------------

def _fmt_products(row: dict) -> str:
    """
    Products.txt  (ApplNo, ProductNo, Form, Strength, DrugName, ActiveIngredient)
    → "PAREDRINE (active ingredient: hydroxyamphetamine hydrobromide) is approved
       under NDA application 000004 as an ophthalmic solution/drops at 1% strength."
    """
    name   = row.get("DrugName", "").strip().title()
    active = row.get("ActiveIngredient", "").strip().title()
    appl   = row.get("ApplNo", "").strip()
    form   = row.get("Form", "").strip().lower()
    strength = row.get("Strength", "").strip()
    parts: list[str] = []
    if name:
        parts.append(name)
        if active and active.lower() != name.lower():
            parts[-1] += f" (active ingredient: {active})"
    if appl:
        parts.append(f"approved under application {appl}")
    if form:
        parts.append(f"formulated as {form}")
    if strength:
        parts.append(f"at a strength of {strength}")
    return " is ".join(parts[:2]) + (", " + ", ".join(parts[2:]) if len(parts) > 2 else "") + "." if parts else ""


def _fmt_applications(row: dict) -> str:
    """
    Applications.txt  (ApplNo, ApplType, SponsorName)
    → "Application 000004 is an NDA submitted by Pharmics."
    """
    appl    = row.get("ApplNo", "").strip()
    atype   = row.get("ApplType", "").strip()
    sponsor = row.get("SponsorName", "").strip().title()
    parts: list[str] = []
    if appl:
        parts.append(f"Application {appl}")
    if atype:
        parts.append(f"a {atype}")
    if sponsor:
        parts.append(f"submitted by {sponsor}")
    if len(parts) == 3:
        return f"{parts[0]} is {parts[1]} {parts[2]}."
    return " ".join(parts) + "." if parts else ""


def _fmt_submissions(row: dict) -> str:
    """
    Submissions.txt  (ApplNo, SubmissionType, SubmissionNo, SubmissionStatus,
                      SubmissionStatusDate, ReviewPriority)
    → "Application 000004 ORIG submission 1 has approval status as of
       1969-07-16 with UNKNOWN review priority."
    """
    appl    = row.get("ApplNo", "").strip()
    stype   = row.get("SubmissionType", "").strip()
    sno     = row.get("SubmissionNo", "").strip()
    status  = row.get("SubmissionStatus", "").strip()
    date    = (row.get("SubmissionStatusDate") or "").strip().split(" ")[0]  # date only
    priority = row.get("ReviewPriority", "").strip()
    notes   = (row.get("SubmissionsPublicNotes") or "").strip()
    s = f"Application {appl} {stype} submission {sno}"
    if status:
        s += f" received status '{status}'"
    if date and date != "0000-00-00":
        s += f" as of {date}"
    if priority and priority.upper() not in ("UNKNOWN", ""):
        s += f" with {priority} review priority"
    if notes:
        s += f". Notes: {notes}"
    return s + "."


def _fmt_appdocs(row: dict) -> str:
    """
    ApplicationDocs.txt  (ApplNo, SubmissionType, SubmissionNo,
                          ApplicationDocsTitle, ApplicationDocsDate)
    → "Application 004782 SUPPL submission 125 document dated 2003-07-28."
    """
    appl  = row.get("ApplNo", "").strip()
    stype = row.get("SubmissionType", "").strip()
    sno   = row.get("SubmissionNo", "").strip()
    title = (row.get("ApplicationDocsTitle") or "").strip()
    date  = (row.get("ApplicationDocsDate") or "").strip().split(" ")[0]
    s = f"Application {appl} {stype} submission {sno} document"
    if title and title != "0":
        s += f": {title}"
    if date and date != "0000-00-00":
        s += f", dated {date}"
    return s + "."


def _fmt_te(row: dict) -> str:
    """
    TE.txt  (ApplNo, ProductNo, TECode)
    → "Application 003444 product 001 has therapeutic equivalence code AA."
    """
    appl = row.get("ApplNo", "").strip()
    prod = row.get("ProductNo", "").strip()
    code = row.get("TECode", "").strip()
    return f"Application {appl} product {prod} has therapeutic equivalence code {code}." if code else ""


def _fmt_marketing_status(row: dict) -> str:
    """
    MarketingStatus.txt  (MarketingStatusID, ApplNo, ProductNo)
    → "Application 000004 product 004 has marketing status ID 3."
    """
    appl   = row.get("ApplNo", "").strip()
    prod   = row.get("ProductNo", "").strip()
    msid   = row.get("MarketingStatusID", "").strip()
    return f"Application {appl} product {prod} has marketing status ID {msid}." if appl else ""


# Lookup tables (ActionTypes_Lookup, MarketingStatus_Lookup, etc.) –
# these are tiny reference tables; render as readable 'key: value' lines.
def _fmt_generic_tsv_row(row: dict) -> str:
    parts = [f"{k}: {v}" for k, v in row.items() if v and str(v).strip()]
    return ". ".join(parts) + "." if parts else ""


# Map filename keywords → row formatter
_FDA_FORMATTERS = {
    "products":          _fmt_products,
    "applications":      _fmt_applications,   # matches Applications.txt but NOT ApplicationDocs
    "applicationdocs":   _fmt_appdocs,
    "submissions":       _fmt_submissions,    # matches Submissions.txt but NOT SubmissionsClass etc.
    "te":                _fmt_te,
    "marketingstatus":   _fmt_marketing_status,
}


def _load_tsv(path: Path) -> str:
    """
    Load a tab-delimited file.
    Routes each row through a per-filename prose formatter so every chunk
    reads as natural-language text rather than raw column dumps.
    Falls back to a generic readable formatter for unrecognised files.
    """
    # Pick the right formatter: match lowercase stem against keyword map.
    # 'applicationdocs' must be checked before 'applications'.
    stem = path.stem.lower().replace("_", "").replace(" ", "").replace("-", "")
    formatter = _fmt_generic_tsv_row  # default
    # Iterate in priority order so 'applicationdocs' beats 'applications'
    for keyword, fn in _FDA_FORMATTERS.items():
        if stem.startswith(keyword) or keyword in stem:
            formatter = fn
            break

    records: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sentence = formatter(row)
            if sentence:
                records.append(sentence)
    return "\n".join(records)


def _load_txt(path: Path) -> str:
    """
    Load a .txt file.  Auto-detects tab-delimited FDA files and routes them
    to the TSV loader; otherwise reads as plain text.
    """
    if _is_tsv(path):
        return _load_tsv(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _sdf_fields_to_prose(fields: dict[str, str]) -> str:
    """
    Convert a dict of SDF annotation fields for one molecule into a
    natural-language sentence.

    Example output:
        "Bivalirudin (DrugBank ID: DB00006, CAS: 128270-60-0, UNII: TN9BEX005G)
         is also known as: Bivalirudina; Bivalirudinum."
    """
    name     = fields.get("COMMON_NAME", "").strip()
    db_id    = fields.get("DRUGBANK_ID", "").strip()
    cas      = fields.get("CAS_NUMBER", "").strip()
    unii     = fields.get("UNII", "").strip()
    synonyms = fields.get("SYNONYMS", "").strip()
    accession = fields.get("SECONDARY_ACCESSION_NUMBERS", "").strip()

    # Build identity clause: "Name (DrugBank ID: X, CAS: Y, UNII: Z)"
    id_parts: list[str] = []
    if db_id:
        id_parts.append(f"DrugBank ID: {db_id}")
    if cas:
        id_parts.append(f"CAS: {cas}")
    if unii:
        id_parts.append(f"UNII: {unii}")

    sentence = name if name else "Unknown compound"
    if id_parts:
        sentence += f" ({', '.join(id_parts)})"

    # Remaining known fields folded into prose
    extra_parts: list[str] = []
    if synonyms:
        extra_parts.append(f"also known as {synonyms}")
    if accession:
        extra_parts.append(f"with accession numbers {accession}")

    # Any additional fields not handled above
    known = {"COMMON_NAME", "DRUGBANK_ID", "CAS_NUMBER", "UNII",
             "SYNONYMS", "SECONDARY_ACCESSION_NUMBERS"}
    for key, val in fields.items():
        if key not in known and val.strip():
            # Format field name: "MOLECULAR_FORMULA" → "molecular formula"
            readable_key = key.replace("_", " ").lower()
            extra_parts.append(f"{readable_key} {val.strip()}")

    if extra_parts:
        sentence += " is " + ", ".join(extra_parts)

    return sentence + "."


def _load_sdf(path: Path) -> str:
    """
    Parse an MDL V2000 SDF file and extract ONLY the data annotation blocks,
    discarding all atom coordinate and bond tables entirely.

    The SDF format:
        [molecule name line]
        [counts line + atom block + bond block]   ← numeric tables, discarded
        M  END
        > <FIELD_NAME>                             ← annotation header
        field value                                ← kept
        (blank line)
        $$$$                                       ← record delimiter

    Each molecule's annotation fields are converted to a natural-language
    prose sentence via _sdf_fields_to_prose().  Molecules are separated by
    blank lines so the text splitter can keep each one together when possible.
    """
    records: list[str] = []
    current_fields: dict[str, str] = {}
    current_field_name: str = ""
    in_annotation = False  # True after M  END, until $$$$

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip("\n").strip()

            if stripped == "$$$$":
                # End of a molecule record – convert fields to prose and save
                if current_fields:
                    records.append(_sdf_fields_to_prose(current_fields))
                current_fields = {}
                current_field_name = ""
                in_annotation = False
                continue

            if stripped == "M  END":
                # Everything before this is coordinate/bond data – start collecting
                in_annotation = True
                continue

            if not in_annotation:
                continue

            # Annotation block: field header line e.g.  > <DRUGBANK_ID>
            if stripped.startswith("> <") and stripped.endswith(">"):
                current_field_name = stripped[3:-1]  # strip '> <' and trailing '>'
                continue

            # Field value line
            if current_field_name and stripped:
                current_fields[current_field_name] = stripped
                current_field_name = ""  # SDF has one value per field block

    # Flush any trailing record (file may not end with $$$$)
    if current_fields:
        records.append(_sdf_fields_to_prose(current_fields))

    return "\n\n".join(records)


def _load_pubmed_xml(path: Path) -> str:
    """
    Stream a MEDLINE PubmedArticleSet XML file using iterparse so the entire
    150-200 MB file is never loaded into memory at once.

    Extracts ONLY:
        <ArticleTitle>   – the title of the article
        <AbstractText>   – one or more abstract paragraphs (with optional Label)

    Author names, journal metadata, dates, MeSH terms, etc. are intentionally
    skipped – they add noise for semantic retrieval of drug/medical content.

    Output: one paragraph per article, separated by blank lines:
        "Title: Formate assay in body fluids…
         Abstract: [BACKGROUND] This study… [METHODS] We used…"
    """
    records: list[str] = []
    title: str = ""
    abstract_parts: list[str] = []

    # iterparse streams the file; we clear each PubmedArticle element after
    # processing to keep memory usage flat regardless of file size.
    for event, elem in ET.iterparse(str(path), events=("end",)):
        tag = elem.tag  # PubMed XML has no namespace prefix

        if tag == "ArticleTitle":
            title = (elem.text or "").strip()

        elif tag == "AbstractText":
            text = (elem.text or "").strip()
            if text:
                label = elem.get("Label", "")
                abstract_parts.append(f"[{label}] {text}" if label else text)

        elif tag == "PubmedArticle":
            # End of one article – build its text record and reset state
            parts: list[str] = []
            if title:
                parts.append(f"Title: {title}")
            if abstract_parts:
                parts.append("Abstract: " + " ".join(abstract_parts))
            if parts:
                records.append("\n".join(parts))

            # Reset state and free memory for this article element
            title = ""
            abstract_parts = []
            elem.clear()

    return "\n\n".join(records)


def _load_json(path: Path) -> str:
    """
    Load a JSON file.  Supports:
        - List[str]   → join items with newlines
        - List[dict]  → extract 'text' / 'content' / 'description' key, or full dump
        - dict        → formatted JSON dump
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        parts: list[str] = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("description")
                    or json.dumps(item)
                )
                parts.append(str(text))
        return "\n".join(parts)

    return json.dumps(data, indent=2)


def load_file(path: Path) -> str | None:
    """
    Dispatch to the appropriate loader based on file extension (and filename).

    Returns the extracted text string, or None if the file type is unsupported.
    """
    ext = path.suffix.lower()
    loaders = {
        ".csv": _load_csv,
        ".txt": _load_txt,   # auto-detects TSV inside
        ".sdf": _load_sdf,
        ".xml": _load_pubmed_xml,
        ".json": _load_json,
    }
    loader = loaders.get(ext)
    if loader is None:
        print(f"[ingest] Skipping unsupported file type: {path.name}")
        return None
    return loader(path)


# ---------------------------------------------------------------------------
# Core ingestion pipeline
# ---------------------------------------------------------------------------

def load_and_chunk_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    Walk `data_dir` recursively, load every supported file, split into chunks,
    and return a list of chunk records.

    Each record is a dict:
        {
            "id"       : str,  # deterministic sha256-based ID
            "text"     : str,  # chunk content
            "source"   : str,  # relative path from data_dir
            "chunk_idx": int,  # position of chunk within the source document
        }
    """
    all_chunks: list[dict] = []

    # Gather all files recursively
    file_paths = sorted(
        p for p in data_dir.rglob("*") if p.is_file()
    )

    print(f"[ingest] Found {len(file_paths)} file(s) under {data_dir}")

    for file_path in file_paths:
        # Skip explicitly excluded files
        if file_path.name in EXCLUDED_FILENAMES:
            print(f"[ingest] Excluded (noise): {file_path.relative_to(data_dir)}")
            continue

        print(f"[ingest] Processing: {file_path.relative_to(data_dir)}")

        raw_text = load_file(file_path)
        if raw_text is None or not raw_text.strip():
            print(f"[ingest]   → empty or unreadable, skipped.")
            continue

        # Split into overlapping chunks
        chunks = _splitter.split_text(raw_text)
        relative_source = str(file_path.relative_to(data_dir))

        for idx, chunk_text in enumerate(chunks):
            # Build a deterministic ID from source + chunk index so re-runs
            # produce stable IDs and do not duplicate vectors in Pinecone.
            raw_id = f"{relative_source}::{idx}"
            chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:32]

            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": relative_source,
                "chunk_idx": idx,
            })

        print(f"[ingest]   → {len(chunks)} chunk(s) produced.")

    print(f"[ingest] Total chunks across all files: {len(all_chunks)}")
    return all_chunks


def ingest(data_dir: Path = DATA_DIR) -> dict:
    """
    Full ingestion pipeline with rolling batches.

    Instead of embedding ALL chunks then upserting (6+ hours on CPU),
    we process INGEST_BATCH_SIZE chunks at a time:
        embed batch → upsert batch → next batch

    Benefits:
    - Vectors are saved to Pinecone incrementally (safe to interrupt)
    - Peak RAM stays flat regardless of total corpus size
    - Progress is visible in real time

    Returns:
        Summary dict with 'files_processed', 'total_chunks', 'upserted_count'.
    """
    INGEST_BATCH_SIZE = 512  # chunks per embed+upsert cycle

    # ── Step 1: Load and chunk all documents ───────────────────────────────
    chunks = load_and_chunk_documents(data_dir)
    if not chunks:
        return {"files_processed": 0, "total_chunks": 0, "upserted_count": 0}

    total = len(chunks)
    total_upserted = 0
    num_batches = (total + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE

    print(f"[ingest] Embedding + upserting {total:,} chunks "
          f"in {num_batches} batches of {INGEST_BATCH_SIZE} …")

    # ── Steps 2–4: Rolling embed → upsert ──────────────────────────────────
    for batch_num, start in enumerate(range(0, total, INGEST_BATCH_SIZE), 1):
        batch = chunks[start : start + INGEST_BATCH_SIZE]

        # Embed this batch
        print(f"[ingest] Batch {batch_num}/{num_batches} "
              f"– embedding {len(batch)} chunks …")
        texts = [c["text"] for c in batch]
        embeddings = embed_documents(texts, batch_size=256)

        # Build Pinecone vector records
        vectors = [
            {
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_idx": chunk["chunk_idx"],
                },
            }
            for chunk, embedding in zip(batch, embeddings)
        ]

        # Upsert this batch into Pinecone immediately
        result = upsert_vectors(vectors)
        total_upserted += result["upserted_count"]
        print(f"[ingest] Batch {batch_num}/{num_batches} done "
              f"– running total: {total_upserted:,}/{total:,} vectors")

    unique_sources = len({c["source"] for c in chunks})
    summary = {
        "files_processed": unique_sources,
        "total_chunks": total,
        "upserted_count": total_upserted,
    }
    print(f"[ingest] Done: {summary}")
    return summary
