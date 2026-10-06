import os
import sys
from pathlib import Path

# Allow this script to run from the project root or from the scripts directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.utils import load_and_process_data
from rag_system.retriever import RetrieverAgent
from rag_system.generator import GeneratorAgent

def debug_fir():
    print("--- 1. Loading Data ---")
    docs = load_and_process_data()
    
    print("\n--- 2. Building Index ---")
    retriever = RetrieverAgent()
    retriever.build_index(docs)
    
    query = "How to file an FIR for theft in India? Step by step"
    print(f"\n--- 3. Retrieving for: '{query}' ---")
    
    # Same k as app
    ret_out = retriever.retrieve(query, k=20)
    chunks = ret_out["chunks"]
    
    print(f"Found {len(chunks)} chunks.")
    print("Top 3 Chunks:")
    for i, c in enumerate(chunks[:3]):
        print(f"[{i+1}] {c['orig_id']} (Score: {c['score']:.4f})")
        print(f"Text: {c['text'][:200]}...\n")
        
    # Check if vital keywords exist
    has_154 = any("154" in c["text"] for c in chunks[:10])
    has_theft = any("theft" in c["text"].lower() for c in chunks[:10])
    print(f"Contains '154' (FIR Section): {has_154}")
    print(f"Contains 'theft': {has_theft}")

    print("\n--- 4. Generating Answer (Mock Mode to check prompt structure, or OpenAI/Groq if key set) ---")
    # We'll use mock just to see the prompt, unless keys are in env
    mode = "mock"
    if os.getenv("OPENAI_API_KEY"): mode = "openai"
    elif os.getenv("GROQ_API_KEY"): mode = "groq"
    
    print(f"Mode: {mode}")
    gen_agent = GeneratorAgent(mode=mode)
    
    # Check prompt text
    prompt = gen_agent._strict_generator_prompt_text(query, chunks[:10])
    print("\nGenerated Prompt Snippet:")
    print(prompt[:500] + "...")
    
    if mode != "mock":
        print("\nAttempting Actual Generation...")
        res = gen_agent.generate(query, chunks[:10])
        print("Result:", res)

if __name__ == "__main__":
    debug_fir()
