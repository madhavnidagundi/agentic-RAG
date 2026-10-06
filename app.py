import streamlit as st
import os
from rag_system.utils import load_and_process_data
from rag_system.retriever import RetrieverAgent
from rag_system.generator import GeneratorAgent
from rag_system.critic import CriticAgent, augment_claims

# Page Config
st.set_page_config(page_title="Legal Agentic RAG", page_icon="⚖️", layout="wide")

st.title("⚖️ Agentic RAG for Indian Law")
st.markdown("Evidence-grounded legal research across the indexed Indian-law materials.")
st.caption("Research support only. Verify every provision against the current authoritative source and apply independent professional judgment.")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys
    openai_key = st.text_input("OpenAI API Key", type="password")
    groq_key = st.text_input("Groq API Key", type="password")
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        
    mode = st.selectbox("Generation Mode", ["openai", "groq"])
    st.caption("Provider keys are used only for the current application process and are never displayed in results.")
    
    st.divider()
    
    st.subheader("Indexing")
    if st.button("Build/Rebuild Index"):
        load_and_process_data.clear()
        st.session_state["index_built"] = False
        st.rerun()

# Initialize State
if "retriever" not in st.session_state:
    st.session_state["retriever"] = RetrieverAgent()
    st.session_state["index_built"] = False

# Load Data & Build Index
if not st.session_state["index_built"]:
    with st.spinner("Loading dataset and building index... this may take a moment."):
        try:
            docs = load_and_process_data()
            st.session_state["retriever"].build_index(docs)
            st.session_state["index_built"] = True

            pdf_count = sum(1 for d in docs if d.get("orig_id", "").endswith(".pdf"))
            dataset_count = len(docs) - pdf_count
            st.success(f"Index built: {len(docs)} evidence chunks.")
            st.info(f"Corpus composition: {pdf_count} PDF chunks | {dataset_count} dataset chunks")
        except (OSError, RuntimeError, ValueError) as exc:
            st.session_state["index_built"] = False
            st.error(f"The evidence index could not be built: {exc}")

# Main Interface
query = st.text_input("Enter your legal query:", placeholder="e.g., What is the punishment for Section 302 IPC?")

if query:
    if not st.session_state["index_built"]:
        st.error("Index not built. Please check the sidebar.")
    else:
        # 1. Retrieve
        try:
            with st.status("🔍 Retrieving Evidence...", expanded=True) as status:
                retriever_out = st.session_state["retriever"].retrieve(query, k=20)
                chunks = retriever_out["chunks"]
                status.write(f"Found {len(chunks)} relevant evidence chunks.")

                with st.expander("Review retrieved evidence"):
                    for i, c in enumerate(chunks[:10]):
                        src = c["orig_id"]
                        page = c.get("metadata", {}).get("page_number")
                        location = f", page {page}" if page else ""
                        icon = "📕" if src.endswith(".pdf") else "📄"
                        st.markdown(f"**[{i + 1}] {icon} {src}{location}** · relevance {c['score']:.3f}")
                        st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
        except (RuntimeError, ValueError) as exc:
            st.error(f"Evidence retrieval failed: {exc}")
            chunks = []
        
        # 2. Generate
        st.subheader("📝 Evidence-grounded research response")
        gen_out = {}
        if not chunks:
            st.warning("No evidence was retrieved. Refine the question or add authoritative source material.")
        elif not ((mode == "openai" and openai_key) or (mode == "groq" and groq_key)):
            st.info(f"Enter a {mode.title()} API key in the sidebar to generate a cited response from the retrieved evidence.")
        else:
            with st.spinner("Generating a cited response..."):
                try:
                    gen_agent = GeneratorAgent(mode=mode)
                    gen_agent.set_api_key(openai_key if mode == "openai" else groq_key)
                    gen_out = gen_agent.generate(query, chunks[:10])
                except (RuntimeError, ValueError) as exc:
                    gen_out = {"message": f"Generation setup failed: {exc}"}

            if gen_out.get("answer_text") == "INSUFFICIENT_EVIDENCE":
                st.warning(gen_out.get("message", "The retrieved evidence is insufficient for a reliable response."))
            elif "answer_text" in gen_out:
                st.markdown(gen_out["answer_text"])
            else:
                st.error(gen_out.get("message", "The system could not produce a valid cited response."))

        # 3. Critique & Augmentation
        if "claim_to_sources" in gen_out and gen_out["claim_to_sources"]:
            st.divider()
            st.subheader("🕵️ Critique & Fact-Checking")
            
            with st.spinner("Critiquing answer..."):
                # Use Groq for critique if in Groq mode and key is present
                critic_mode = "heuristic"
                if mode == "groq" and groq_key:
                    critic_mode = "groq"
                
                critic = CriticAgent(mode=critic_mode, retriever_obj=st.session_state["retriever"])
                if critic_mode == "groq" and groq_key:
                    critic.set_api_key(groq_key)
                    if critic.client:
                        st.info("🔎 Running AI Verifier (Groq/Llama-3)...")
                    else:
                        st.warning("⚠️ Groq client not initialized. Falling back to Heuristic.")
                else:
                    st.info("Running Heuristic Verifier (Rule-based)...")
                    
                crit_out = critic.critique(gen_out, chunks[:10])
                
                # Handle possible error from Groq critic
                if "error" in crit_out:
                    st.error(f"Critic Error: {crit_out.get('message') or crit_out.get('raw')}")
                else: 
                    # Only show metrics if no error
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Factuality Score", f"{crit_out.get('factuality', 0.0):.2f}", help="Percentage of claims supported by evidence.")
                    col2.metric("Coherence", f"{crit_out.get('coherence', 0.0):.2f}", help="Quality of text flow and logic.")
                    col3.metric("Citation Alignment", f"{crit_out.get('citation_alignment', 0.0):.2f}", help="Do citations [1] match the real sources?")
                    
                    if "reasoning" in crit_out:
                        st.info(f"**Critic Reasoning:** {crit_out['reasoning']}")
                    
                    with st.expander("Show Evidence Used for Critique"):
                        st.text(f"The Critic saw the following {len(chunks[:10])} chunks:")
                        for i, c in enumerate(chunks[:10]):
                            st.text(f"[{i+1}] {c['text'][:200]}...")
                    
                    with st.expander("Show Critic Raw Analysis"):
                        st.json(crit_out)
                    
                    if crit_out.get("unsupported_claims"):
                        st.warning("Some claims require legal review before use.")
                        with st.expander(f"Review {len(crit_out['unsupported_claims'])} claim(s) requiring support"):
                            for uc in crit_out["unsupported_claims"]:
                                if isinstance(uc, dict):
                                    st.write(f"- Claim: '{uc.get('span', 'Unknown')}' -> Issue: {uc.get('issue', 'Unknown')}")
                                else:
                                    st.write(f"- Issue: {uc}")

                        try:
                            repaired = augment_claims(gen_out, st.session_state["retriever"], query)
                            with st.expander("Additional evidence located for review"):
                                for claim in repaired["claim_to_sources_augmented"]:
                                    st.markdown(f"**Claim {claim.get('claim_id')}:** {claim.get('span')}")
                                    sources = claim.get("augmented_sources", [])
                                    if not sources:
                                        st.caption("No additional semantically similar evidence was found.")
                                    for source in sources:
                                        pool_chunk = repaired["pool_chunks"][source["pool_index"]]
                                        metadata = pool_chunk.get("metadata", {})
                                        page = metadata.get("page_number")
                                        location = f", page {page}" if page else ""
                                        st.write(f"{pool_chunk.get('orig_id')}{location} · similarity {source['sim']:.2f}")
                                        st.caption(source["snippet"])
                        except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                            st.info(f"Additional evidence search was unavailable: {exc}")

                    else:
                        st.success("All claims appear supported by retrieval context.")


st.markdown("---")
st.caption("Agentic RAG System | Built with Streamlit")
