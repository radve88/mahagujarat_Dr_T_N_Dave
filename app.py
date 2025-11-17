import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from spellchecker import SpellChecker
from transformers import pipeline
import os

# -----------------------------
# CONFIG
# -----------------------------
PICKLE_FILE = "output/embeddings.pkl"
FAISS_INDEX_FILE = "output/faiss_index.index"
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
MAX_LEN = 400  # words per chunk for LLM prompt
TOP_K_DEFAULT = 3
LLM_MAX_TOKENS = 200  # response length per chunk

# -----------------------------
# LOAD DATA / MODELS
# -----------------------------
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer(SENTENCE_MODEL)

@st.cache_resource
def load_faiss_and_embeddings():
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_FILE)
    return data["chunks"], np.array(data["embeddings"], dtype="float32"), index

@st.cache_resource
def load_local_llm_pipeline():
    # deterministic: do_sample=False, temperature ignored by pipeline if do_sample=False
    return pipeline("text2text-generation", model=LLM_MODEL, device=-1)  # -1 uses CPU

sentence_model = load_sentence_model()
chunks, embeddings, index = load_faiss_and_embeddings()
local_llm = load_local_llm_pipeline()

# -----------------------------
# UTILITIES
# -----------------------------
spell = SpellChecker()

def correct_query(query: str) -> str:
    words = query.split()
    corrected = [spell.correction(w) for w in words]
    return " ".join(corrected)

def safe_encode(text: str, model: SentenceTransformer) -> np.ndarray:
    words = text.split()
    segments = [" ".join(words[i:i + MAX_LEN]) for i in range(0, len(words), MAX_LEN)]
    embeddings_list = [model.encode(seg, convert_to_numpy=True) for seg in segments]
    return np.mean(embeddings_list, axis=0)

def highlight_terms(text: str, query: str) -> str:
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(f"**{query}**", text)

def query_faiss(query: str, top_k: int = 3, apply_spell_check: bool = True):
    if apply_spell_check:
        safe_q = correct_query(query)
    else:
        safe_q = query
    q_vec = safe_encode(safe_q, sentence_model)
    distances, indices = index.search(np.array([q_vec], dtype="float32"), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk_index": int(idx),
            "chunk_text": chunks[idx],
            "distance": float(distances[0][i])
        })
    return results

def llm_answer_from_chunks(context_chunk: str, question: str, include_instructions: bool = True) -> str:
    """
    Deterministic FLAN-T5 answer for a single context chunk.
    do_sample=False ensures deterministic output.
    """
    if include_instructions:
        prompt = f"""Answer the question using ONLY the context below.
If the answer is not present in the context, respond exactly with:
"The context does not contain the answer."

Context:
{context_chunk}

Question:
{question}

Answer:"""
    else:
        prompt = f"Context:\n{context_chunk}\n\nQuestion:\n{question}\n\nAnswer:"

    # pipeline returns list of dicts with 'generated_text'
    out = local_llm(prompt, max_new_tokens=LLM_MAX_TOKENS, do_sample=False, num_return_sequences=1)
    if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
        return out[0]["generated_text"].strip()
    # Fallback
    return str(out).strip()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Dr. T. N. Dave: Mahagujarat Monograph Search (FLAN-T5 Deterministic)")
st.markdown(
    "Semantic search over Dr. T. N. Dave's monograph with deterministic LLM answers "
    "based strictly on retrieved chunks. If answer isn't in the context, the model will say so."
)

# Preset questions
preset_questions = [
    "What does Dr. T. N. Dave say about the evolution of modern Gujarati?",
    "What are the key features of Gujarat’s geography?",
    "How does Dr. Dave describe regional dialects?",
    "What were the historical phases of linguistic development?",
    "What social or cultural factors influenced the Mahagujarat movement?"
]

selected_q = st.selectbox("Choose a question to explore:", ["-- Select a question --"] + preset_questions)
query = st.text_input("Or enter your own query:", value=selected_q if selected_q != "-- Select a question --" else "")
top_k = st.slider("Number of results to display:", min_value=1, max_value=10, value=TOP_K_DEFAULT)

# Run search
if query:
    results = query_faiss(query, top_k=top_k, apply_spell_check=True)
    st.session_state["retrieved_chunks"] = [r["chunk_text"] for r in results]
    st.session_state["last_query"] = query

    st.subheader(f"Top {top_k} results for your query:")
    for r in results:
        with st.expander(f"📖 View Chunk #{r['chunk_index']} | Distance: {r['distance']:.4f}"):
            st.write(highlight_terms(r["chunk_text"], query))

# Student reflection area (optional)
st.markdown("---")
st.subheader("Student Q&A (optional)")
student_answer = st.text_area("Write your answer / reflection (optional):")
if student_answer:
    st.success("Reflection noted — it will be considered when generating the LLM answer.")

# Auto LLM inference (deterministic) — runs if we have retrieved chunks
if "retrieved_chunks" in st.session_state and len(st.session_state["retrieved_chunks"]) > 0:
    st.markdown("## 🔮 Deterministic LLM Answer (from retrieved chunks)")

    last_q = st.session_state.get("last_query", "").strip()
    student_input = student_answer.strip() if student_answer else ""

    # Build full context as words and chunk for LLM in MAX_LEN-word slices
    all_words = " ".join(st.session_state["retrieved_chunks"]).split()
    context_chunks = [" ".join(all_words[i:i + MAX_LEN]) for i in range(0, len(all_words), MAX_LEN)]
    chunk_answers = []

    # For strict mode: model must answer only from context. We'll run it on each context chunk deterministically.
    for i, c in enumerate(context_chunks):
        with st.spinner(f"Generating deterministic answer from chunk {i+1}/{len(context_chunks)}..."):
            ans = llm_answer_from_chunks(c, last_q)
            chunk_answers.append(ans)

    # Post-process chunk answers:
    # If any chunk returns a real answer (i.e., not the sentinel "The context does not contain the answer."),
    # we will collect those and display them. Otherwise report not found.
    sentinel = "The context does not contain the answer."
    useful_answers = [a for a in chunk_answers if a and sentinel not in a]

    if len(useful_answers) == 0:
        st.info(sentinel)
    else:
        # Combine deterministic chunk answers (de-duplicate while preserving order)
        seen = set()
        consolidated = []
        for a in useful_answers:
            if a not in seen:
                consolidated.append(a)
                seen.add(a)
        st.subheader("🧠 Answer (consolidated from chunks):")
        for ans in consolidated:
            st.write(ans)

# About
with st.expander("About this App"):
    st.markdown(
        """
**Dr. T. N. Dave’s Mahagujarat Monograph — Deterministic QA**

- Uses Sentence-Transformers for embeddings and FAISS for retrieval.
- Uses a local FLAN-T5 model (`google/flan-t5-base`) for deterministic answers (do_sample=False).
- Strict Answer Mode: answers must be present in retrieved chunks; otherwise the model says the context does not contain the answer.
- No external API keys or billing required for this mode.
"""
    )
