import time
import json
import numpy as np
from typing import List, Dict, Any, Optional

class RAGEvaluationSuite:
    """
    Implements a comprehensive RAG Evaluation Suite.
    
    METRICS:
    - Retrieval: Recall@K, Precision@K, MRR (Mean Reciprocal Rank)
    - Grounding: Faithfulness, Hallucination Rate, Citation Alignment
    - Legal Quality: Correctness, Completeness, Reasoning Validity
    - Safety: Refusal Quality (Correctly refusing irrelevant queries)
    - System: Latency, Cost (Tokens - estimated)
    """
    
    def __init__(self, retriever, generator, judge_client, judge_model="llama-3.1-8b-instant"):
        """
        Args:
            retriever: The RAG RetrieverAgent object.
            generator: The RAG GeneratorAgent object.
            judge_client: An initialized Groq or OpenAI client to act as the "Judge".
            judge_model: Model name for the judge (e.g. 'llama-3.1-8b-instant').
        """
        self.retriever = retriever
        self.generator = generator
        self.client = judge_client
        self.judge_model = judge_model

    # ==========================================
    # 1. RETRIEVAL METRICS
    # ==========================================
    def evaluate_retrieval(self, query: str, ground_truth_doc_ids: List[str], k: int = 10) -> Dict[str, float]:
        """
        Calculates Recall, Precision, and MRR against a list of known relevant Doc IDs.
        """
        ret_out = self.retriever.retrieve(query, k=k)
        retrieved_chunks = ret_out["chunks"]
        retrieved_ids = [c.get("orig_id", "") for c in retrieved_chunks]
        
        # 1. Recall@K
        # How many of the relevant docs did we find?
        hits = sum(1 for doc in ground_truth_doc_ids if doc in retrieved_ids)
        recall = hits / len(ground_truth_doc_ids) if ground_truth_doc_ids else 0.0
        
        # 2. Precision@K
        # How many of our retrieved items were relevant?
        precision = hits / k
        
        # 3. MRR (Mean Reciprocal Rank)
        # What was the rank of the first relevant item?
        mrr = 0.0
        for i, doc in enumerate(retrieved_ids):
            if doc in ground_truth_doc_ids:
                mrr = 1.0 / (i + 1)
                break
                
        return {
            "Recall@K": recall, 
            "Precision@K": precision, 
            "MRR": mrr,
            "retrieved_chunks": retrieved_chunks
        }

    # ==========================================
    # 2. GENERATION & SAFETY METRICS (LLM JUDGE)
    # ==========================================
    def evaluate_generation_quality(self, query: str, context_chunks: List[dict], answer: str, gold_answer: Optional[str] = None) -> Dict[str, float]:
        """
        Uses an LLM Judge to score Faithfulness, Legal Correctness, and Refusal Quality.
        """
        # Prepare context for the judge
        context_text = "\n".join([f"{i+1}. {c['text'][:300]}" for i,c in enumerate(context_chunks)])
        
        prompt = f"""
        YOU ARE AN EXPERT LEGAL EVALUATOR.
        
        TASK: Rate the AI Answer based on the Query, Context, and (optional) Ground Truth.
        
        DATA:
        - Query: {query}
        - Context: {context_text}
        - AI Answer: {answer}
        - Ground Truth: {gold_answer if gold_answer else 'N/A'}
        
        SCORING CRITERIA (0.0 to 1.0):
        1. FAITHFULNESS: Is the answer derived PURELY from the provided Context? (1.0 = Fully grounded, 0.0 = Hallucinated/Outside Knowledge).
        2. LEGAL_CORRECTNESS: Is the legal reasoning valid? (If Ground Truth is provided, does it match? If not, is it logically sound based on Context?).
        3. REFUSAL_QUALITY: 
           - If Context is IRRELEVANT, did the AI say "INSUFFICIENT_EVIDENCE"? (Score 1.0).
           - If Context IS RELEVANT, did the AI provided a good answer? (Score 1.0).
           - Score 0.0 if it hallucinated an answer when it should have refused, or refused when context was present.
        4. CITATION_ALIGNMENT: Do the citations (e.g. [1]) actually match the logic? (1.0 = Perfect).
        
        RETURN JSON format:
        {{
            "faithfulness": float,
            "legal_correctness": float,
            "refusal_quality": float,
            "citation_alignment": float,
            "reasoning": "short explanation"
        }}
        """
        
        try:
            resp = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role":"user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            metrics = json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"Eval Error: {e}")
            metrics = {"faithfulness": 0.0, "legal_correctness": 0.0, "refusal_quality": 0.0, "citation_alignment": 0.0}
            
        # Derived Metrics
        metrics["hallucination_rate"] = 1.0 - metrics.get("faithfulness", 0.0)
        return metrics

    # ==========================================
    # 3. RUN FULL BENCHMARK
    # ==========================================
    def run_benchmark(self, test_set: List[Dict]) -> List[Dict]:
        """
        Runs the full evaluation loop on a test set.
        current_test_set = [
            {"query": "...", "ground_truth_doc_ids": ["doc1.pdf"], "ground_truth_answer": "..."}
        ]
        """
        results = []
        print(f"Running Evaluation on {len(test_set)} queries...")
        
        total_latency = 0
        
        for i, item in enumerate(test_set, 1):
            q = item["query"]
            gold_ids = item.get("ground_truth_doc_ids", [])
            gold_ans = item.get("ground_truth_answer", None)
            
            print(f"evaluating {i}/{len(test_set)}: {q[:40]}...")
            
            # A. Latency & Retrieval
            t0 = time.time()
            ret_metrics = self.evaluate_retrieval(q, gold_ids)
            
            # B. Generation
            # We use the chunks found in step A
            gen_out = self.generator.generate(q, ret_metrics["retrieved_chunks"])
            latency = time.time() - t0
            total_latency += latency
            
            answer_text = gen_out.get("answer_text", str(gen_out))
            
            # C. Quality Grading
            qual_metrics = self.evaluate_generation_quality(
                q, ret_metrics["retrieved_chunks"], 
                answer_text, 
                gold_ans
            )
            
            # Compile Row
            row = {
                "query": q,
                "latency_sec": round(latency, 2),
                **{k:v for k,v in ret_metrics.items() if k != "retrieved_chunks"}, # Exclude raw chunks from CSV/DF
                **qual_metrics
            }
            results.append(row)
            
        print(f"\nEvaluation Complete. Avg Latency: {total_latency/len(test_set):.2f}s")
        return results

# ==================
# EXAMPLE USAGE
# ==================
# if __name__ == "__main__":
#     # Assuming you have instantiated 'retriever', 'wrapper.generator', and a 'client' (Groq/OpenAI) 
#     evaluator = RAGEvaluationSuite(retriever, wrapper.generator, verifier.client)
#     
#     test_data = [
#         {
#             "query": "What is the punishment for Section 302?", 
#             "ground_truth_doc_ids": ["ipc-bare-act.pdf"], 
#             "ground_truth_answer": "Death or imprisonment for life and fine."
#         }
#     ]
# 
#     results = evaluator.run_benchmark(test_data)
#     print(results)
