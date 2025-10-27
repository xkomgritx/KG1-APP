import streamlit as st
import os
from huggingface_hub import snapshot_download # เพิ่มส่วนนี้เข้ามา
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- การตั้งค่า ---
st.set_page_config(page_title="AI Research Assistant", layout="wide")

# --- ส่วนที่เพิ่มเข้ามา: ฟังก์ชันดาวน์โหลดไฟล์จาก Hugging Face ---
@st.cache_resource(show_spinner="กำลังดาวน์โหลดงานวิจัย...")
def download_papers_from_hf():
    """
    ดาวน์โหลดไฟล์ทั้งหมดจาก Hugging Face Dataset repository
    และคืนค่า path ของโฟลเดอร์ที่เก็บไฟล์
    """
    # *** แก้ไขตรงนี้: ใส่ชื่อ Dataset ของคุณ ***
    repo_id = "xkomgritx/my-research-papers-dataset" # เช่น "john-doe/my-research-papers-dataset"
    
    local_dir = "./my_research_papers_from_hf"
    if not os.path.exists(local_dir):
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    return local_dir

# --- ส่วนที่เหลือของโค้ดจะคล้ายเดิม แต่จะใช้ path ที่ได้จากฟังก์ชันด้านบน ---

# ... (โค้ดส่วน UI และการตั้งค่า API Key เหมือนเดิม) ...

# --- ฟังก์ชันหลักสำหรับสร้าง RAG Pipeline (ปรับปรุงเล็กน้อย) ---
@st.cache_resource(show_spinner="กำลังประมวลผลงานวิจัย...")
def load_and_index_papers(papers_path): # รับ path เป็น argument
    # ... (โค้ดข้างในฟังก์ชันนี้เหมือนเดิมทุกประการ) ...
    # 1. โหลดเอกสาร PDF ทั้งหมด
    loader = PyPDFDirectoryLoader(papers_path)
    docs = loader.load()
    # ... (ส่วนที่เหลือเหมือนเดิม) ...
    # (ค��ดลอกโค้ดจากฟังก์ชัน load_and_index_papers เดิมมาวางที่นี่ได้เลย)
    if not os.path.exists(papers_path) or not os.listdir(papers_path):
        return None
    loader = PyPDFDirectoryLoader(papers_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    prompt = ChatPromptTemplate.from_template("""
    You are a world-class AI research assistant...
    Context: {context}
    Question: {input}
    Answer (in the same language as the question):""")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


# --- ส่วนหลักของแอป ---
# ตั้งค่า API Key
with st.sidebar:
    st.header("การตั้งค่า")
    openai_api_key = st.text_input("กรอก OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key

st.title("👨‍🔬 AI Research Assistant (via Streamlit Cloud)")

if not openai_api_key:
    st.info("กรุณากรอก OpenAI API Key ของคุณในแถบด้านข้างเพื่อเริ่มต้น")
    st.stop()

# 1. ดาวน์โหลดไฟล์ก่อน
papers_folder_path = download_papers_from_hf()

# 2. ส่ง path ที่ได้ไปให้ฟังก์ชันสร้าง RAG chain
rag_chain = load_and_index_papers(papers_folder_path)

# ... (ส่วน UI ที่เหลือสำหรับแชทเหมือนเดิมทุกประการ) ...
if rag_chain:
    st.success("AI พร้อมตอบคำถามเกี่ยวกับงานวิจัยของคุณแล้ว!")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("ถามเกี่ยวกับงานวิจัยของคุณ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นคว้าและเรียบเรียงคำตอบ..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                with st.expander("📚 ดูแหล่งข้อมูลที่ใช้อ้างอิง"):
                    sources = {doc.metadata.get('source', 'N/A') for doc in response["context"]}
                    st.write(f"คำตอบนี้สังเคราะห์มาจากเปเปอร์: {', '.join(sources)}")
        st.session_state.messages.append({"role": "assistant", "content": answer})
