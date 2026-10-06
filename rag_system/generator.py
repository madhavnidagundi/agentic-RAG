import os
import json
import re
from typing import List, Dict, Any, Optional

try:
    import openai
except ImportError:
    openai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

class GeneratorAgent:
    """
    GeneratorAgent: produces structured JSON answers.
    Modes: 'mock', 'openai', 'groq'
    """

    def __init__(self, mode: str = "mock", model_name: str = "gpt-4o-mini", temperature: float = 0.0, max_len_chars: int = 3000, max_retries: int = 2):
        assert mode in ("mock", "openai", "groq"), "mode must be 'mock', 'openai', or 'groq'"
        self.mode = mode
        self.model_name = model_name
        self.temperature = temperature
        self.max_len_chars = max_len_chars
        self.max_retries = max_retries
        self.client = None

        if self.mode == "openai":
            if openai is None: raise RuntimeError("openai SDK not installed.")
            key = os.getenv("OPENAI_API_KEY")
            # For Streamlit, we might want to allow setting key later or read from st.secrets if available
            # But for now, let's just warn or init if env var present
            if key:
                openai.api_key = key
            
        if self.mode == "groq":
            if Groq is None: raise RuntimeError("groq SDK not installed. Run pip install groq")
            key = os.getenv("GROQ_API_KEY")
            if key:
                self.client = Groq(api_key=key)
            if "gpt" in self.model_name:
                self.model_name = "llama-3.1-8b-instant"

    def set_api_key(self, key: str):
        if self.mode == "openai" and openai:
            openai.api_key = key
        elif self.mode == "groq" and Groq:
            self.client = Groq(api_key=key)

    def _build_evidence_block(self, chunks: List[Dict[str,Any]], snippet_chars: int = 400) -> str:
        lines = []
        for i, c in enumerate(chunks, start=1):
            metadata = c.get("metadata", {})
            src = c.get("orig_id") or metadata.get("source", f"doc_{i}")
            page = metadata.get("page_number")
            location = f", page {page}" if page else ""
            snippet = (c.get("text") or "")[:snippet_chars].replace("\n"," ").strip()
            lines.append(f"{i}. {snippet} — Source: {src}{location}")
        return "\n".join(lines)

    def _strict_generator_prompt_text(self, question: str, chunks: List[Dict[str,Any]]):
        evidence_block = self._build_evidence_block(chunks, snippet_chars=600)
        prompt = (
            "You are an evidence-grounded legal research assistant. Your task is to answer the user's question using ONLY the provided evidence.\n"
            "DO NOT use any internal knowledge. If the answer is not in the evidence, say INSUFFICIENT_EVIDENCE.\n\n"
            f"User Question: {question}\n\n"
            "EVIDENCE (Read carefully):\n"
            f"{evidence_block}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. **TOPIC MATCHING**: Check if the Evidence actually matches the TOPIC of the Question. \n"
            "   - Example: If Question is 'Criminal Law' (Theft, FIR) but Evidence is 'Tax/Civil Law' (Income Tax, Forms), do NOT use it.\n"
            "   - Return INSUFFICIENT_EVIDENCE if there is a Domain Mismatch.\n"
            "2. **NO OUTSIDE KNOWLEDGE**: You are forbidden from using any knowledge outside the Evidence block. However, you MAY use logical inference to connect facts within the evidence.\n"
            "3. **NO LEGAL CONCLUSIONS**: For factual scenarios, identify only the provisions and limits supported by the evidence. Do not predict a verdict, outcome, charge, or defence unless it is expressly stated in the evidence.\n"
            "4. **VERBATIM PREFERENCE**: When citing legal sections, punishments, or definitions, use the exact wording from the evidence.\n"
            "5. **STRICT CITATION**: Every factual sentence must end with a citation like [1] or [1,2].\n"
            "6. **NULL HYPOTHESIS**: If the evidence is insufficient or IRRELEVANT (Topic Mismatch), return a JSON where \"answer_text\" is exactly \"INSUFFICIENT_EVIDENCE\". Do not make up an answer.\n"
            "7. **FORMAT**: Return ONLY a JSON object with this structure:\n"
            "   {\n"
            "     \"answer_text\": \"...answer string... OR 'INSUFFICIENT_EVIDENCE'\",\n"
            "     \"claim_to_sources\": [\n"
            "        {\"claim_id\": 1, \"span\": \"exact sentence from answer\", \"sources\": [1]}\n"
            "     ],\n"
            "     \"generation_confidence\": 1.0\n"
            "   }\n"
            f"8. Max Length: {self.max_len_chars} chars.\n"
        )
        return prompt

    def _extract_json(self, text: str) -> Optional[Dict[str,Any]]:
        text = text.strip()
        if text == "INSUFFICIENT_EVIDENCE":
            return {"special": "INSUFFICIENT_EVIDENCE", "answer_text": "Insufficient evidence found.", "claim_to_sources": [], "generation_confidence": 0.0}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'(\{[\s\S]*\})', text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    return None
            return None

    def _call_openai(self, prompt_text: str) -> str:
        messages = [{"role":"system","content":"You are a careful assistant. Output only JSON."},
                    {"role":"user","content":prompt_text}]
        resp = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1500
        )
        return resp["choices"][0]["message"]["content"].strip()

    def _call_groq(self, prompt_text: str) -> str:
        if not self.client:
            return '{"error": "GROQ_API_KEY not set"}'
        messages = [{"role":"system","content":"You are a careful assistant. Output only JSON."},
                    {"role":"user","content":prompt_text}]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1500
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _insufficient_evidence(message: str = "The retrieved materials do not contain sufficient evidence to answer this question.") -> Dict[str, Any]:
        return {
            "answer_text": "INSUFFICIENT_EVIDENCE",
            "claim_to_sources": [],
            "generation_confidence": 0.0,
            "message": message,
        }

    @staticmethod
    def _validate_claims(payload: Dict[str, Any], chunk_count: int) -> bool:
        if not isinstance(payload, dict):
            return False
        claims = payload.get("claim_to_sources")
        if not isinstance(claims, list) or not claims:
            return False
        for claim in claims:
            if not isinstance(claim, dict):
                return False
            sources = claim.get("sources")
            if not isinstance(claim.get("span"), str) or not isinstance(sources, list):
                return False
            if not sources or any(not isinstance(source, int) or not 1 <= source <= chunk_count for source in sources):
                return False
        return True

    def generate(self, question: str, chunks: List[Dict[str,Any]]) -> Dict[str,Any]:
        if not question or not question.strip():
            return self._insufficient_evidence("Enter a legal research question to continue.")
        if not chunks:
            return self._insufficient_evidence()

        prompt_text = self._strict_generator_prompt_text(question, chunks)
        
        if self.mode == "mock":
             return {"answer_text": "Mock mode disabled.", "claim_to_sources": [], "generation_confidence": 0.0}

        elif self.mode in ("openai", "groq"):
            for _ in range(self.max_retries):
                try:
                    raw = self._call_openai(prompt_text) if self.mode == "openai" else self._call_groq(prompt_text)
                except Exception as exc:
                    return self._insufficient_evidence(f"Generation provider error: {exc}")
                
                parsed = self._extract_json(raw)
                if isinstance(parsed, dict):
                    ans_text = parsed.get("answer_text", "").strip()
                    if parsed.get("special") == "INSUFFICIENT_EVIDENCE" or "INSUFFICIENT_EVIDENCE" in ans_text:
                        return self._insufficient_evidence()
                    if not self._validate_claims(parsed, len(chunks)):
                        prompt_text = "Your response must map every claim to valid evidence indexes. Return ONLY valid JSON.\n" + prompt_text
                        continue
                    
                    parsed.setdefault("generation_confidence", 0.5)
                    parsed["raw"] = raw
                    return parsed
                
                prompt_text = "Previous response was invalid JSON. Return ONLY valid JSON.\n" + prompt_text
            
            return self._insufficient_evidence("The model did not return a valid, fully cited evidence-grounded response.")
        
        return {}
