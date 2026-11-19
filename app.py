
import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

st.set_page_config(page_title="ResumeRAG AI", page_icon="🤖", layout="wide")
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🤖 ResumeRAG Assistant")
st.subheader("Your Personal AI Resume Chatbot")
st.write("Upload your resume and ask questions using AI.")

with st.sidebar:
    st.title("ResumeRAG")
    st.write("📄 Resume Analysis")
    st.write("🧠 Smart Search")
    st.write("💬 AI Chat")
    st.divider()
    st.caption("Powered by Google Gemini")

def extract_text(pdf):
    return " ".join(p.extract_text() or "" for p in PdfReader(pdf).pages)

def chunk_text(text, size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0,len(words),size-overlap)]

def embed(texts):
    r = client.models.embed_content(model="gemini-embedding-001",contents=texts)
    return np.array([e.values for e in r.embeddings])

st.divider()
st.header("📤 Upload Your Resume")
pdf = st.file_uploader("Choose PDF Resume", type="pdf")

if pdf and st.button("🚀 Process Resume"):
    text = extract_text(pdf)
    if not text.strip():
        st.error("No readable text found!")
        st.stop()
    st.session_state.chunks = chunk_text(text)
    st.session_state.embeddings = embed(st.session_state.chunks)
    st.session_state.processed = True
    st.session_state.messages = []
    st.success("Resume processed successfully!")

if st.session_state.get("processed", False):
    st.divider()
    st.header("💬 AI Chat Assistant")

    for m in st.session_state.get("messages", []):
        st.chat_message(m["role"]).write(m["content"])

    if question := st.chat_input("Ask about your resume..."):
        st.session_state.messages.append({"role":"user","content":question})
        st.chat_message("user").write(question)

        if question.lower().strip() in ["hi","hello","hey","hii"]:
            answer = "Hello! 👋 I am your Resume Assistant. Ask me anything!"
        else:
            q = embed([question])[0]
            docs = st.session_state.embeddings
            scores = np.dot(docs,q)/(np.linalg.norm(docs,axis=1)*np.linalg.norm(q)+1e-10)
            top = np.argsort(scores)[-7:][::-1]
            context = "\n".join(st.session_state.chunks[i] for i in top)

            prompt = f"""You are a friendly Resume AI Assistant.
Answer using only the resume context.
Never invent personal information.
If information is missing, say it is not mentioned in the resume.
Keep answers simple and interview-friendly.
Resume Context: {context}
Question: {question}
Answer:"""

            r = client.models.generate_content(model="gemini-3.5-flash-lite",contents=prompt)
            answer = r.text or "Unable to generate answer."

        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role":"assistant","content":answer})

st.divider()
st.caption("❤️ ResumeRAG | Powered by Google Gemini")
