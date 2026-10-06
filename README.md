# Agentic RAG for Indian Legal Research

This project is a Streamlit application that helps legal professionals research questions against a selected corpus of Indian legal materials. It uses an agentic Retrieval-Augmented Generation (RAG) workflow to retrieve relevant text, draft an evidence-grounded response, and check whether each drafted claim is supported by the retrieved evidence.

> **Important:** This is a legal-research and drafting aid, not legal advice. It does not replace a lawyer's independent analysis, verification of the current law, or professional judgment. Statutes may be amended, repealed, superseded, or interpreted differently by courts.

## What it does

The application accepts a legal question, such as *"What is the punishment under Section 302 IPC?"*, and follows this workflow:

1. **Load the corpus** – Reads the PDFs in `data/source_documents/` and attempts to load `viber1/indian-law-dataset` from Hugging Face.
2. **Prepare legal text** – Extracts PDF text and divides source material into overlapping chunks so that surrounding legal context is retained.
3. **Retrieve evidence** – Uses hybrid retrieval:
   - Dense vector search (SentenceTransformers + FAISS) for semantic relevance.
   - BM25 keyword search for exact legal terminology, statute names, and section references.
   - A weighted fusion of the two result sets.
4. **Draft a response** – Uses OpenAI or Groq, when configured, to answer from the retrieved chunks and produce claim-to-source mappings.
5. **Verify claims** – A critic evaluates cited claims with either heuristics (token overlap and embedding similarity) or a separate Groq verifier.
6. **Show evidence and quality signals** – The interface displays retrieved excerpts plus factuality, coherence, citation-alignment, and potential unsupported-claim results.

## How this can support a lawyer

Use it to accelerate early-stage research, issue spotting, and drafting—not to make final legal conclusions. Useful lawyer-facing applications include:

- Locating potentially relevant provisions across the available Acts and dataset.
- Checking whether an initial draft proposition is supported by the retrieved material.
- Surfacing exact source excerpts to review before using a point in advice, a pleading, or a submission.
- Identifying claims that need more research when the critic marks them as partially supported or unsupported.
- Comparing terminology across the Constitution, IPC, CrPC, BNSS, and dataset records included in the local index.

### Suggested professional review process

1. State the facts and jurisdiction precisely in the question.
2. Read the retrieved excerpts in full; do not rely on a score alone.
3. Open and verify the authoritative, current version of every cited Act, rule, notification, and judgment.
4. Confirm whether newer law applies—for example, whether a provision has been replaced or changed.
5. Check controlling and recent case law, procedural posture, limitation, local rules, and factual distinctions outside this corpus.
6. Treat the generated answer as a research lead or draft, then revise it using the lawyer's own analysis.

## Included source material

The repository currently includes PDFs for:

- The Constitution of India
- Indian Penal Code (IPC)
- Code of Criminal Procedure (CrPC), 1973
- Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023

The application may also download the Hugging Face dataset at runtime. Network access and availability of that dataset affect whether it is included in the index.

## Run locally

Install the Python dependencies required by the source files, configure at least one model provider key, then start Streamlit:

```powershell
$env:OPENAI_API_KEY = "your-key"  # or set GROQ_API_KEY
streamlit run app.py
```

You can also enter an OpenAI or Groq key in the app sidebar and choose the corresponding generation mode. On the first run, the app builds the index; use **Build/Rebuild Index** after changing source materials.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and orchestration of retrieve → generate → verify. |
| `rag_system/retriever.py` | FAISS and BM25 hybrid retriever. |
| `rag_system/generator.py` | Evidence-grounded LLM generation and claim/source output parsing. |
| `rag_system/critic.py` | Heuristic or Groq-based claim verification and evidence augmentation utilities. |
| `rag_system/utils.py` | PDF extraction, chunking, dataset loading, and metadata helpers. |
| `data/source_documents/` | Curated PDF source materials indexed by the application. |
| `data/evaluation/` | Benchmark outputs. |
| `evaluation/` | Evaluation utilities. |
| `scripts/` | Local debugging and diagnostic scripts. |
| `notebooks/` | Exploratory notebooks. |

## Interpreting the results

- **Retrieved chunks** are leads, not automatically authoritative citations.
- **Factuality** reflects the critic's assessment against the retrieved context only; it does not establish legal correctness.
- **Citation alignment** indicates whether a generated claim points to a supplied evidence index; it is not a Bluebook/Indian citation check.
- **Insufficient evidence** means the system declined to produce a legal answer outside the retrieved evidence.

## Limitations

- The system's answer is bounded by its corpus, chunking quality, retrieval quality, and model output.
- The included statutes and data may be incomplete or outdated.
- It does not guarantee amendment status, case-law currency, or jurisdiction-specific treatment. Page numbers identify the extracted PDF page and should be checked against the original document.
- Model outputs and automated fact-checking can still be wrong. Never use the output as the sole basis for legal advice or a filing.
