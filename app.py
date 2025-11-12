import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from spellchecker import SpellChecker
from transformers import pipeline
import torch

# ---------- CONFIG ----------
PICKLE_FILE = "output/embeddings.pkl"
FAISS_INDEX_FILE = "output/faiss_index.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- LOAD DATA ----------
@st.cache_resource
def load_data():
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_FILE)
    model = SentenceTransformer(MODEL_NAME)
    return data["chunks"], data["embeddings"], index, model

chunks, embeddings, index, model = load_data()

# ---------- SPELL CHECKER ----------
spell = SpellChecker()

def correct_query(query):
    words = query.split()
    corrected = [spell.correction(w) for w in words]
    return " ".join(corrected)

# ---------- QUERY FUNCTION ----------
def query_faiss(query, top_k=3, apply_spell_check=True):
    if apply_spell_check:
        query = correct_query(query)
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_vec, dtype="float32"), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk_index": idx,
            "chunk_text": chunks[idx],
            "distance": distances[0][i]
        })
    return results

# ---------- LOAD FREE HUGGING FACE MODEL ----------
@st.cache_resource
def load_local_llm():
    try:
        llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",   # or 'flan-t5-small' for lighter version
            device=0 if torch.cuda.is_available() else -1
        )
        return llm
    except Exception as e:
        st.error(f"Error loading local model: {e}")
        return None

llm = load_local_llm()

# ---------- INFERENCE FUNCTION ----------
def infer_with_local_llm(query, retrieved_chunks):
    if not llm:
        return "LLM not available."

    context = "\n\n".join([r["chunk_text"] for r in retrieved_chunks])
    prompt = f"""
You are assisting in understanding Dr. T. N. Dave's 1948 monograph *The Language of Maha Gujarat*.

Context:
{context}

Question:
{query}

Provide a concise academic answer based only on the above context.
"""
    try:
        response = llm(prompt, max_new_tokens=200)
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error during inference: {e}"

# ---------- HIGHLIGHT FUNCTION ----------
def highlight_terms(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(f"**{query}**", text)

# ---------- STREAMLIT APP ----------
st.title("Dr. T. N. Dave: Mahagujarat Monograph Search")
st.markdown("""
This app allows researchers to explore Dr. T. N. Dave's Mahagujarat monograph.
It extracts text from the PDF, creates embeddings, and performs semantic search 
so you can find relevant sections efficiently.
""")

# Top-k slider
top_k = st.slider("Number of results to display:", min_value=1, max_value=10, value=3)

query = st.text_input("Enter your search query:")

if query:
    results = query_faiss(query, top_k=top_k, apply_spell_check=True)
    st.subheader(f"Top {top_k} results:")
    for r in results:
        st.markdown(f"**Chunk #{r['chunk_index']} | Distance: {r['distance']:.4f}**")
        st.write(highlight_terms(r["chunk_text"], query))
        st.markdown("---")

    # ---- Inference section ----
    if st.button("🧠 Generate AI Summary / Answer"):
        st.subheader("LLM Inference Result:")
        answer = infer_with_local_llm(query, results)
        st.success(answer)

# ---------- ABOUT SECTION ----------
with st.expander("About this App"):
    st.markdown("""
**Dr. T. N. Dave's Mahagujarat Monograph**  

Dr. T. N. Dave's monograph is a comprehensive study of the language and dialects of Gujarat,
covering historical, regional, and modern developments. It is a seminal work for anyone
interested in linguistics, history, and regional studies of India.

This app was created to help researchers, students, and language enthusiasts
search and explore the content of the monograph efficiently.  

This app is developed with love and devotion and tribute by me Rashmikant Dave an AI Architect and Developer 
to my Grandfather Dr T N Dave the Author of the Mahagujarat Study and Monograph and dedicate to all my family
and extended family of Dr T N Dave

Respects to all other learned teachers and scholars of the Gujarati Language

**Features:**  
- Full-text search with semantic understanding via embeddings.  
- Query-term highlighting for easy reference.  
- Adjustable number of results (top-k).  
- Integrated free Hugging Face model for summarization/inference.
""")

