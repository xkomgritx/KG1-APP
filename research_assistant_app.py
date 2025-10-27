import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- ตั้งค่าหน้าเว็บ Streamlit ---
st.set_page_config(page_title="AI Research Assistant", layout="wide")

# --- ตั้งค่า OpenAI API Key ผ่าน Sidebar ---
with st.sidebar:
    st.header("การตั้งค่า")
    openai_api_key = st.text_input("กรอก OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key

# --- ส่วนหัวของแอปพลิเคชัน ---
st.title("👨‍🔬 AI Research Assistant")
st.write("ผู้ช่วย AI ที่เชี่ยวชาญในงานวิจัยของคุณ ถามคำถามเกี่ยวกับเปเปอร์ของคุณได้เลย")

# --- ฟังก์ชันหลักสำหรับสร้าง RAG Pipeline ---
@st.cache_resource(show_spinner="กำลังประมวลผลงานวิจัยของคุณ...")
def load_and_index_papers(papers_path):
    """
    โหลดไฟล์ PDF ทั้งหมด, ทำ Indexing, และสร้าง RAG chain
    """
    if not os.path.exists(papers_path) or not os.listdir(papers_path):
        return None

    # 1. โหลดเอกสาร PDF ทั้งหมด
    loader = PyPDFDirectoryLoader(papers_path)
    docs = loader.load()

    # 2. ตัดแบ่งเอกสาร (สำคัญมากสำหรับเปเปอร์ยาวๆ)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # เพิ่มขนาด chunk สำหรับเนื้อหาทางวิชาการ
        chunk_overlap=400
    )
    splits = text_splitter.split_documents(docs)

    # 3. สร้าง Vector Store (ดัชนีสำหรับค้นหา)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # ใช้ embedding model ที่ดีขึ้น
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # 4. สร้าง Prompt Template ที่ออกแบบมาสำหรับงานวิจัย
    prompt = ChatPromptTemplate.from_template("""
    You are a world-class AI research assistant, specializing in the work of this specific author.
    Your task is to answer questions accurately and concisely based ONLY on the provided research papers.
    Synthesize information from multiple papers if the question requires it.
    When you answer, be precise and refer to methodologies, results, or discussions found in the context.

    Context from the papers:
    {context}

    Question: {input}

    Answer (in the same language as the question):""")

    # 5. สร้าง RAG Chain
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1) # Temp ต่ำเพื่อความแม่นยำ
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # ดึง context มา 5 ชิ้น เพื่อให้มีข้อมูลประกอบการสังเคราะห์มากขึ้น
    )
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- ส่วนติดต่อกับผู้ใช้ (User Interface) ---
if not openai_api_key:
    st.info("กรุณากรอก OpenAI API Key ของคุณในแถบด้านข้างเพื่อเริ่มต้น")
    st.stop()

# โหลด RAG pipeline
rag_chain = load_and_index_papers("./my_research_papers/")

if rag_chain is None:
    st.error("ไม่พบไฟล์งานวิจัยในโฟลเดอร์ 'my_research_papers' กรุณาเพิ่มไฟล์ PDF ก่อน")
    st.stop()

st.success("AI พร้อมตอบคำถามเกี่ยวกับงานวิจัยของคุณแล้ว!")

# สร้าง session state เพื่อเก็บประวัติการแชท
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการแชท
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# รับอินพุตจากผู้ใช้
if prompt := st.chat_input("ถามเกี่ยวกับงานวิจัยของคุณ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นคว้าและเรียบเรียงคำตอบ..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            
            # แสดงคำตอบ
            st.markdown(answer)
            
            # แสดงแหล่งอ้างอิง
            with st.expander("📚 ดูแหล่งข้อมูลที่ใช้อ้างอิง"):
                sources = {doc.metadata.get('source', 'N/A') for doc in response["context"]}
                st.write(f"คำตอบนี้สังเคราะห์มาจากเปเปอร์: {', '.join(sources)}")
                st.json([doc.metadata for doc in response["context"]])


    st.session_state.messages.append({"role": "assistant", "content": answer})
