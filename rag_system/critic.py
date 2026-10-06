import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

try:
    from groq import Groq, BadRequestError
except ImportError:
    Groq = None
    BadRequestError = None

class CriticAgent:
    """
    CriticAgent:
    - mode: 'heuristic' or 'groq'
    - heuristic uses embeddings + token overlap
    - groq uses Llama-3.1 on Groq API to classify claims as Supported / Partial / Unsupported.
    """

    def __init__(self, mode="heuristic", retriever_obj=None,
                 groq_model="llama-3.1-8b-instant",
                 sim_threshold=0.68):
        assert mode in ("heuristic", "groq")
        self.mode = mode
        self.retriever = retriever_obj
        self.sim_threshold = sim_threshold
        self.client = None
        self.groq_model = groq_model

        if mode == "groq":
            key = os.getenv("GROQ_API_KEY")
            if key and Groq:
                self.client = Groq(api_key=key)

    def set_api_key(self, key: str):
        if self.mode == "groq" and Groq:
            self.client = Groq(api_key=key)

    # -------------- TOKENIZER --------------
    @staticmethod
    def _simple_tokens(text: str) -> List[str]:
        txt = (text or "").lower()
        txt = re.sub(r"[^\w\s]", " ", txt)
        return [t for t in txt.split() if len(t) > 2]

    # -------------- EMBEDDING --------------
    def _embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if self.retriever and hasattr(self.retriever, "model"):
            try:
                embs = self.retriever.model.encode(texts, convert_to_numpy=True)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                return embs / norms
            except Exception as e:
                print("[Critic] Embedding error:", e)
                return None
        return None

    # -------------- HEURISTIC SCORE FOR ONE CLAIM --------------
    def heuristic_score_claim(self, span, sources, chunks):
        tokens_claim = set(self._simple_tokens(span))
        best_sim, best_overlap = 0.0, 0.0
        matched = []

        for s in sources:
            if not isinstance(s, int): continue
            if not (1 <= s <= len(chunks)):
                continue
            chunk_text = chunks[s-1]["text"]
            tokens_chunk = set(self._simple_tokens(chunk_text))
            overlap = len(tokens_claim & tokens_chunk) / max(1, len(tokens_claim))
            best_overlap = max(best_overlap, overlap)

            emb_pair = self._embed([span, chunk_text])
            if emb_pair is not None:
                try:
                    sim = float(cosine_similarity([emb_pair[0]], [emb_pair[1]])[0][0])
                except (ValueError, IndexError):
                    sim = 0.0
            else:
                sim = 0.0
            best_sim = max(best_sim, sim)

            if overlap > 0.1 or sim >= self.sim_threshold:
                matched.append({"evidence_index": s, "overlap": overlap, "sim": sim})

        # classify
        if best_sim >= self.sim_threshold or best_overlap >= 0.30:
            status = "supported"
        elif best_overlap >= 0.10 or best_sim >= (self.sim_threshold - 0.15):
            status = "partially_supported"
        else:
            status = "unsupported"

        return {
            "status": status,
            "best_sim": best_sim,
            "best_overlap": best_overlap,
            "matched_sources": matched
        }

    # -------------- HEURISTIC CRITIC --------------
    def _heuristic(self, draft, chunks):
        answer_text = draft.get("answer_text", "")
        claim_map = draft.get("claim_to_sources", [])
        per_claim, unsupported = [], []

        for cm in claim_map:
            cid = cm["claim_id"]
            span = cm["span"]
            sources = cm["sources"]
            res = self.heuristic_score_claim(span, sources, chunks)
            per_claim.append({
                "claim_id": cid,
                "span": span,
                "sources": sources,
                **res
            })
            if res["status"] != "supported":
                unsupported.append({
                    "claim_id": cid,
                    "span": span,
                    "issue": res["status"],
                    "suggested_edit": "Rephrase or weaken unsupported claim."
                })

        # Aggregate factuality
        if per_claim:
            supported = sum(1 for p in per_claim if p["status"] == "supported")
            partial = sum(1 for p in per_claim if p["status"] == "partially_supported")
            factuality = (supported + 0.5 * partial) / len(per_claim)
        else:
            factuality = 0.5

        coherence = 1.0 if len(answer_text) > 80 else 0.6
        citation_alignment = (len(per_claim) - len(unsupported)) / len(per_claim) if per_claim else 0.0

        return {
            "factuality": float(factuality),
            "coherence": float(coherence),
            "citation_alignment": float(citation_alignment),
            "unsupported_claims": unsupported,
            "per_claim": per_claim
        }

    # -------------- GROQ CRITIC --------------
    def _groq_critic(self, draft, chunks):
        if not self.client:
             return {"error": "api_fail", "message": "Groq Client not initialized. Cannot run AI Critic."}

        evidence_block = ""
        for i, c in enumerate(chunks, start=1):
            snippet = c["text"][:350].replace("\n"," ")
            src = c.get("orig_id")
            evidence_block += f"{i}. {snippet} — Source: {src}\n"

        claims = draft.get("claim_to_sources", [])
        claims_text = "\n".join(
            [f"{c['claim_id']}. {c['span']} // cited: {c['sources']}" for c in claims]
        ) or draft.get("answer_text", "")

        prompt = (
            "EVIDENCE:\n" + evidence_block + "\n\n"
            "CLAIMS:\n" + claims_text + "\n\n"
            "TASK:\n"
            "Analyze the text and claims against the evidence ONLY. Do not use external truth.\n"
            "Classify each claim as Supported / PartiallySupported / Unsupported.\n"
            "Calculate specific, granular scores (e.g., 0.83, 0.92) based on the evidence match. Do not output round numbers unless necessary.\n"
            " - Factuality: % of claims supported directly by evidence sentences.\n"
            " - Coherence: Text flow and grammatical quality (0.0-1.0).\n"
            " - Citation Alignment: Are the citations pointing to the correct source IDs? (0.0-1.0).\n"
            "1. **ESCAPE QUOTES**: If your reasoning contains quotes, escape them (e.g. \\\"text\\\").\n"
            "2. **NO MARKDOWN**: Do not wrap in ```json.\n"
            "Return a STRICT JSON object:\n"
            "{\n"
            '  "reasoning": "string explanation",\n'
            '  "factuality": float,\n'
            '  "coherence": float,\n'
            '  "citation_alignment": float,\n'
            '  "unsupported_claims": [ {"span": "text", "issue": "reason"} ],\n'
            '  "per_claim": [ {"claim_id":int, "status":str, "reason":str} ]\n'
            "}\n"
        )

        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are a critical, precise factual evaluator. Output only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )
            except BadRequestError as e:
                # Fallback: strict JSON failed (likely validation error), try loose mode
                print(f"[Critic] JSON Mode failed: {e}. Retrying without strict mode...")
                response = self.client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are a critical, precise factual evaluator. Output only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
            raw = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks if present
            if "```" in raw:
                raw = re.sub(r"```json", "", raw)
                raw = re.sub(r"```", "", raw)
                raw = raw.strip()

            # Try parse
            try: 
                return json.loads(raw)
            except json.JSONDecodeError as e:
                # Fallback: try to find the first { and last }
                m = re.search(r"(\{[\s\S]*\})", raw)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except json.JSONDecodeError as inner_e:
                        return {"error": "parse_fail", "message": f"Malformed JSON from AI: {inner_e}", "raw": raw}
                return {"error": "parse_fail", "message": "No JSON object found in response", "raw": raw}

        except Exception as e:
            return {"error": "api_fail", "message": f"Critic System Error: {str(e)}"}

    def critique(self, draft, chunks):
        if self.mode == "heuristic":
            return self._heuristic(draft, chunks)
        else:
            return self._groq_critic(draft, chunks)

def augment_claims(gen_out: Dict[str,Any], retriever, query: str,
                   expand_k: int = 50, sim_threshold: float = 0.60,
                   max_sources_per_claim: int = 4) -> Dict[str,Any]:
    
    expanded = retriever.retrieve(query, k=expand_k)
    pool_chunks = expanded["chunks"]
    pool_texts = [c["text"] for c in pool_chunks]

    claims = gen_out.get("claim_to_sources", [])
    if not claims:
        out = dict(gen_out)
        out["claim_to_sources_augmented"] = []
        out["pool_chunks"] = pool_chunks
        return out

    claim_texts = [c["span"] for c in claims]

    try:
        claim_embs = retriever.model.encode(claim_texts, convert_to_numpy=True)
        pool_embs = retriever.model.encode(pool_texts, convert_to_numpy=True)
    except Exception as e:
        print("[augment] Embedding failure:", e)
        out = dict(gen_out)
        out["claim_to_sources_augmented"] = []
        out["pool_chunks"] = pool_chunks
        return out

    def _norm(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n==0] = 1e-9
        return a / n
    claim_embs = _norm(claim_embs)
    pool_embs = _norm(pool_embs)

    sims = cosine_similarity(claim_embs, pool_embs)

    augmented_claims = []
    for i, claim in enumerate(claims):
        sim_row = sims[i]
        idxs_sorted = np.argsort(-sim_row)
        chosen = []
        for idx in idxs_sorted:
            sim_score = float(sim_row[idx])
            if sim_score < sim_threshold: break
            chosen.append({
                "pool_index": int(idx),
                "pool_doc_id": pool_chunks[idx]["doc_id"],
                "sim": sim_score,
                "snippet": pool_chunks[idx]["text"][:300]
            })
            if len(chosen) >= max_sources_per_claim: break
        
        augmented_claims.append({
            **claim,
            "augmented_sources": chosen
        })

    out = dict(gen_out)
    out["claim_to_sources_augmented"] = augmented_claims
    out["pool_chunks"] = pool_chunks
    return out
