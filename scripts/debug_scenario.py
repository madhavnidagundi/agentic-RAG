import os
import json
import sys
from pathlib import Path

# Allow this script to run from the project root or from the scripts directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.utils import load_and_process_data
from rag_system.retriever import RetrieverAgent
from rag_system.generator import GeneratorAgent

def debug_scenario():
    print("--- 1. Loading Data & Index (Mocking Index Build to save time if possible, else full build) ---")
    # In a real debug script we might want to load a saved index, but here we rebuild fast or expect it to take a moment.
    # We will use the standard load function.
    docs = load_and_process_data()
    
    retriever = RetrieverAgent()
    retriever.build_index(docs)
    
    query = "The appellant was attacked by a person who came to his clinic armed with a pistol and shot him first. The appellant snatched the pistol and shot the assailant dead."
    print(f"\n--- 2. Retrieving for: '{query}' ---")
    
    ret_out = retriever.retrieve(query, k=20)
    chunks = ret_out["chunks"]
    
    print(f"Found {len(chunks)} chunks.")
    print("Top 3 Retrieval Matches:")
    for i, c in enumerate(chunks[:3]):
        print(f"[{i+1}] {c['orig_id']} (Score: {c['score']:.4f})")
        print(f"Text: {c['text'][:200]}...\n")
        
    print("\n--- 3. Testing Generation Path ---")
    gen_agent = GeneratorAgent(mode="mock") # Use mock to see which prompt is constructed logic-wise
    
    # We want to see if the STRICT prompt would likely trigger INSUFFICIENT_EVIDENCE
    # effectively simulating the decision logic.
    
    print("Analyzing Retrieval Relevance...")
    # Check key terms in top chunks
    keywords = ["private defense", "96", "97", "99", "100", "provocation"]
    hits = []
    for c in chunks[:10]:
        text_lower = c["text"].lower()
        found = [k for k in keywords if k in text_lower]
        if found:
            hits.append((c['orig_id'], found))
            
    if hits:
        print(f"RAG PATH LIKELY: Found relevant legal sections in chunks: {hits[:3]}...")
    else:
        print("INSUFFICIENT_EVIDENCE PATH LIKELY: No direct keywords for self-defense found in top chunks.")

    # Also checking if we can run a real generation if API key exists
    if os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"):
        mode = "openai" if os.getenv("OPENAI_API_KEY") else "groq"
        print(f"\n--- 4. Running ACTUAL Generation ({mode}) ---")
        gen_agent = GeneratorAgent(mode=mode)
        
        res = gen_agent.generate(query, chunks[:10])
        print("\nFinal Result Keys:", res.keys())
        if res.get("answer_text") == "INSUFFICIENT_EVIDENCE":
            print(">> CONFIRMED: Insufficient evidence; no uncited answer generated.")
        else:
            print(">> CONFIRMED: Used RAG (Retrieved Chunks)")
            print("Citations present:", res.get("claim_to_sources"))
            
if __name__ == "__main__":
    debug_scenario()
