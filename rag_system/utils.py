import re
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "source_documents"

# 1. Sliding Window Chunker
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 400) -> List[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size.")
    if not text: return []
    if len(text) <= chunk_size: return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# 2. Field Merger (SMART VERSION)
def format_legal_data(ex: dict) -> str:
    # Explicitly check for common structure
    instruction = ex.get("instruction") or ex.get("question") or ex.get("Instruction") or ""
    input_text = ex.get("input") or ex.get("context") or ex.get("Input") or ""
    output = ex.get("output") or ex.get("answer") or ex.get("Output") or ex.get("Response") or ""

    # If we found structured data, use it
    if instruction or output:
        return f"Q: {instruction}\n\nContext: {input_text}\n\nLaw/Answer: {output}".strip()

    # FALLBACK: If keys don't match, just combine ALL string values in the row
    all_text = []
    for k, v in ex.items():
        if isinstance(v, str) and len(v) > 5:
            all_text.append(f"{k}: {v}")
    return "\n\n".join(all_text)

# 3. Category Assigner
def assign_category_v2(text: str, orig_id: str, metadata: dict) -> str:
    txt = (text or "").lower()
    oid = (orig_id or "").lower()
    # explicit patterns
    if "constitution" in txt or re.search(r"\barticle\s+\d+\b", txt) or "article " in txt:
        return "constitution"
    if re.search(r"\bsection\s*\d{1,4}\b", txt) or "ipc" in txt or "indian penal" in txt:
        return "ipc"
    if "crpc" in txt or "criminal procedure" in txt or "procedure" in txt:
        return "crpc"
    # orig_id hints
    if "constitution" in oid or "const" in oid:
        return "constitution"
    if "ipc" in oid or "penal" in oid:
        return "ipc"
    # head sniff
    head = (text or "")[:400].lower()
    if "article " in head:
        return "constitution"
    if "section " in head:
        return "ipc"
    return "other"

def normalize_docs_with_category(docs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    normalized = []
    for d in docs:
        meta = dict(d.get("metadata", {}))
        cat = assign_category_v2(d.get("text",""), d.get("orig_id",""), meta)
        meta["category"] = cat
        normalized.append({
            "doc_id": d.get("doc_id"),
            "orig_id": d.get("orig_id"),
            "text": d.get("text"),
            "metadata": meta
        })
    return normalized

import os
import pypdf

# ... (existing imports)

def process_pdfs(data_dir: str | Path = SOURCE_DOCUMENTS_DIR) -> List[Dict[str, Any]]:
    docs = []
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Source-document directory not found: {data_path}")
    pdf_files = [path for path in data_path.iterdir() if path.suffix.lower() == ".pdf"]
    
    print(f"[PDF-Processor] Found {len(pdf_files)} PDFs in {data_dir}")
    
    for path in pdf_files:
        filename = path.name
        try:
            print(f"[PDF-Processor] Parsing {filename}...")
            reader = pypdf.PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                txt = page.extract_text()
                if not txt or not txt.strip():
                    continue
                for chunk_number, chunk in enumerate(chunk_text(txt, chunk_size=1200, overlap=400), start=1):
                    docs.append({
                        "doc_id": f"pdf_{filename}_page_{page_number}_chunk_{chunk_number}",
                        "orig_id": filename,
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "category": "pdf_act",
                            "page_number": page_number,
                            "chunk_index": chunk_number,
                        },
                    })
        except Exception as e:
            print(f"[PDF-Processor] Failed to read {filename}: {e}")
            
    print(f"[PDF-Processor] Created {len(docs)} chunks from PDFs.")
    return docs

@st.cache_data
def load_and_process_data():
    all_docs: List[Dict] = []
    
    # 1. Load curated PDFs from the project source-documents directory.
    pdf_docs = process_pdfs()
    all_docs.extend(pdf_docs)

    # 2. Load HuggingFace Dataset when its optional dependency is available.
    if load_dataset is None:
        st.warning("The optional 'datasets' package is not installed; indexing local PDF sources only.")
        return all_docs

    try:
        print("[docs-creator] Loading dataset...")
        ds = load_dataset("viber1/indian-law-dataset")
        split = "train" if "train" in ds else list(ds.keys())[0]
        dataset = ds[split]

        print(f"[docs-creator] Processing {len(dataset)} examples...")

        hf_docs = []
        for i, ex in enumerate(dataset):
            full_text = format_legal_data(ex)
            if len(full_text) < 15:
                continue

            orig_id = f"viber1/indian-law-dataset/{i}"
            text_chunks = chunk_text(full_text, chunk_size=1200, overlap=400)

            for chunk_idx, chunk_content in enumerate(text_chunks):
                hf_docs.append({
                    "doc_id": f"{orig_id}_chunk_{chunk_idx}",
                    "orig_id": orig_id,
                    "text": chunk_content,
                    "metadata": {"source": orig_id, "chunk_index": chunk_idx}
                })
        
        normalized_hf = normalize_docs_with_category(hf_docs)
        print(f"[docs-creator] Success! Created {len(normalized_hf)} chunks from HF dataset.")
        all_docs.extend(normalized_hf)

        return all_docs

    except Exception as e:
        print(f"[docs-creator] Error: {e}")
        st.error(f"Failed to load dataset: {e}")
        # Return at least the PDFs if HF fails
        return all_docs
