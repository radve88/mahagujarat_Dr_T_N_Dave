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
from openai import OpenAI


# -----------------------------
# STREAMLIT SECRETS → OPENAI KEY
# -----------------------------
client = OpenAI(api_key=st.secrets["general"]["API_KEY"])


# -----------------------------
# CONFIG
# -----------------------------
PICKLE_FILE = "output/embeddings.pkl"
FAISS_INDEX_FILE = "output/faiss_index.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 400


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_resource
def load_data():
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_FILE)
    model = SentenceTransformer(MODEL_NAME)
    return data["chunks"], data["embeddings"], index, model


chunks, embeddings, index, model = load_data()


# -----------------------------
# OPTIONAL LOCAL MODEL
# -----------------------------
@st.cache_resource
def load_local_llm():
    try:
        qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
        return qa_pipeline
    except Exception as e:
        st.warning(f"Local model load failed: {e}")
        return None


local_llm = load_local_llm()


# -----------------------------
# OPTIONAL HUGGINGFACE INFERENCE
# -----------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def query_huggingface_inference(prompt):
    if not HF_API_TOKEN:
        return "[HF Token Missing]"

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}

    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    return f"Error {response.status_code}: {response.text}"


# -----------------------------
# SPELL CHECKER
# -----------------------------
spell = SpellChecker()

def correct_query(query):
    words = query.split()
    corrected = [spell.correction(w) for w in words]
    return " ".join(corrected)


# -----------------------------
# SAFE EMBEDDING FUNCTION
# -----------------------------
def safe_encode(text, model):
    words = text.split()
    segments = [" ".join(words[i:i + MAX_LEN]) for i in range(0, len(words), MAX_LEN)]
    embeddings = [model.encode(seg, convert_to_numpy=True) for seg in segments]
    return np.mean(embeddings, axis=0)


# -----------------------------
# FAISS RETRIEVAL
# -----------------------------
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


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Dr. T. N. Dave: Mahagujarat Monograph Search")

st.markdown("""
Semantic search + LLM question answering based on  
**Dr. T. N. Dave’s Mahagujarat Monograph**.
""")


# -----------------------------
# PRESET QUESTIONS
# -----------------------------
preset_questions = [
    "What does Dr. T. N. Dave say about the evolution of modern Gujarati?",
    "What are the key features of Gujarat’s geography?",
    "How does Dr. Dave describe regional dialects?",
    "What were the historical phases of linguistic development?",
    "What social or cultural factors influenced the Mahagujarat movement?"
]

selected_q = st.selectbox("Choose a question:", ["-- Select --"] + preset_questions)
query = st.text_input("Or type your own query:", value=selected_q if selected_q != "-- Select --" else "")

top_k = st.slider("Number of results:", min_value=1, max_value=10, value=3)


# -----------------------------
# RUN SEMANTIC SEARCH
# -----------------------------
if query:
    results = query_faiss(query, top_k=top_k)

    st.session_state["retrieved_chunks"] = [r["chunk_text"] for r in results]
    st.session_state["last_query"] = query

    st.subheader(f"Top {top_k} results")

    for r in results:
        with st.expander(f"Chunk #{r['chunk_index']} (Distance: {r['distance']:.4f})"):
            st.write(r["chunk_text"])


# -----------------------------
# LLM ANSWERING USING OPENAI
# -----------------------------
if "retrieved_chunks" in st.session_state and len(st.session_state["retrieved_chunks"]) > 0:

    st.markdown("## 🔮 LLM Answer")

    input_query = st.session_state.get("last_query", "")
    context_text = "\n\n".join(st.session_state["retrieved_chunks"])

    final_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "The answer is not in the provided text."

Context:
{context_text}

Question:
{input_query}

Answer:
"""

    try:
        with st.spinner("Generating answer..."):
            response = client.responses.create(
                model="gpt-4o-mini",
                input=final_prompt
            )
            llm_answer = response.output_text

        st.write(llm_answer)

    except Exception as e:
        st.error(f"LLM Error: {str(e)}")


# -----------------------------
# STUDENT REFLECTION MODE
# -----------------------------
st.markdown("---")
st.subheader("🧠 Reflection")
user_answer = st.text_area("Write your understanding:")

if user_answer:
    st.success("Reflection saved. Continue learning!")


# -----------------------------
# ABOUT SECTION
# -----------------------------
with st.expander("About"):
    st.markdown("""
**Mahagujarat Monograph Search Tool**  
- Semantic search using Sentence Transformers  
- FAISS vector indexing  
- OpenAI LLM answer generation  
- Built for researchers and students  
""")
   



       



