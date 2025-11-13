import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from spellchecker import SpellChecker
import os
import requests
from transformers import pipeline


# ---------- CONFIG ----------
PICKLE_FILE = "output/embeddings.pkl"
FAISS_INDEX_FILE = "output/faiss_index.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 400  # chunk long text safely

# ---------- LOAD DATA ----------
@st.cache_resource
def load_data():
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_FILE)
    model = SentenceTransformer(MODEL_NAME)
    return data["chunks"], data["embeddings"], index, model

chunks, embeddings, index, model = load_data()
@st.cache_resource
def load_local_llm():
    try:
        model_name = "google/flan-t5-base"
        qa_pipeline = pipeline("text2text-generation", model=model_name)
        return qa_pipeline
    except Exception as e:
        st.warning(f"Local model load failed: {e}")
        return None

local_llm = load_local_llm()

# ---------- OPTIONAL: Hugging Face Inference API ----------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Add token in Streamlit Secrets (for Cloud)
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def query_huggingface_inference(prompt):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        else:
            return str(data)
    else:
        return f"[Error {response.status_code}] {response.text}"

# ---------- SPELL CHECKER ----------
spell = SpellChecker()

def correct_query(query):
    words = query.split()
    corrected = [spell.correction(w) for w in words]
    return " ".join(corrected)

# ---------- SAFE CHUNKER ----------
def safe_encode(text, model):
    # Break long queries into smaller parts
    words = text.split()
    segments = [" ".join(words[i:i + MAX_LEN]) for i in range(0, len(words), MAX_LEN)]
    embeddings = [model.encode(seg, convert_to_numpy=True) for seg in segments]
    return np.mean(embeddings, axis=0)  # average embedding

# ---------- QUERY FUNCTION ----------
def query_faiss(query, top_k=3, apply_spell_check=True):
    if apply_spell_check:
        query = correct_query(query)
    query_vec = safe_encode(query, model)
    distances, indices = index.search(np.array([query_vec], dtype="float32"), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk_index": idx,
            "chunk_text": chunks[idx],
            "distance": distances[0][i]
        })
    return results

# ---------- HIGHLIGHT FUNCTION ----------
def highlight_terms(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(f"**{query}**", text)

# ---------- STREAMLIT APP ----------
st.title("Dr. T. N. Dave: Mahagujarat Monograph Search")
st.markdown("""
This app helps researchers and students explore Dr. T. N. Dave’s *Mahagujarat Monograph* 
through semantic search and interactive study features.
""")

# ---------- PRESET QUESTIONS ----------
st.subheader("Explore Key Questions:")
preset_questions = [
    "What does Dr. T. N. Dave say about the evolution of modern Gujarati?",
    "What are the key features of Gujarat’s geography?",
    "How does Dr. Dave describe regional dialects?",
    "What were the historical phases of linguistic development?",
    "What social or cultural factors influenced the Mahagujarat movement?"
]

selected_q = st.selectbox("Choose a question to explore:", ["-- Select a question --"] + preset_questions)
query = st.text_input("Or enter your own search query:", value=selected_q if selected_q != "-- Select a question --" else "")

top_k = st.slider("Number of results to display:", min_value=1, max_value=10, value=3)

# ---------- RUN SEARCH ----------
if query:
    results = query_faiss(query, top_k=top_k, apply_spell_check=True)
    st.subheader(f"Top {top_k} results for your query:")
    for r in results:
        with st.expander(f"📖 View Chunk #{r['chunk_index']} | Distance: {r['distance']:.4f}"):
            st.write(highlight_terms(r["chunk_text"], query))

# ---------- INTERACTIVE Q&A ----------
st.markdown("---")
st.subheader("🧠 Student Q&A Mode")
st.markdown("Try answering the questions above based on what you read. Type your thoughts below:")
user_answer = st.text_area("Your answer:")
if user_answer:
    st.success("✅ Great — your reflection has been noted! Try refining it based on the relevant chunks.")

# ---------- ABOUT SECTION ----------
with st.expander("About this App"):
    st.markdown("""
**Dr. T. N. Dave’s Mahagujarat Monograph**  

A seminal study of Gujarat’s language, dialects, and historical evolution — now accessible through
semantic search and contextual exploration.

**Features:**  
- Intelligent search across the entire monograph.  
- Spell correction for cleaner queries.  
- Optional “View Chunk” mode for readability.  
- Built-in academic Q&A practice for deeper learning.  
""")

