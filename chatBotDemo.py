import streamlit as st
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="C++ RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Theme Toggle
# --------------------------------------------------
theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

if theme:
    bg = "#0E1117"
    chat_bg = "#262730"
    text_color = "white"
else:
    bg = "#FFFFFF"
    chat_bg = "#F1F3F6"
    text_color = "black"

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown(f"""
<style>

body {{
    background-color:{bg};
}}

.chat-container {{
    padding:15px;
    border-radius:12px;
    background-color:{chat_bg};
    margin-bottom:10px;
}}

.user-msg {{
    text-align:right;
    color:{text_color};
}}

.bot-msg {{
    text-align:left;
    color:{text_color};
}}

.title {{
    text-align:center;
    font-size:42px;
    font-weight:700;
}}

.subtitle {{
    text-align:center;
    font-size:18px;
}}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<p class="title">🤖 C++ RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask anything about C++ fundamentals</p>', unsafe_allow_html=True)

# --------------------------------------------------
# Load ENV
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------
@st.cache_resource
def load_vectorstore():

    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db

db = load_vectorstore()

# --------------------------------------------------
# Chat Memory
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for role, msg, sources in st.session_state.messages:

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(msg)

    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg)

            if sources:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(sources):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(src)

# --------------------------------------------------
# Chat Input
# --------------------------------------------------
query = st.chat_input("Ask something about C++...")

if query:

    # Show user message
    with st.chat_message("user", avatar="👤"):
        st.write(query)

    docs = db.similarity_search(query, k=3)

    sources = [doc.page_content for doc in docs]

    answer = ""
    for doc in docs:
        answer += doc.page_content + " "

    # Bot message container
    with st.chat_message("assistant", avatar="🤖"):

        message_placeholder = st.empty()

        # Streaming typing effect
        streamed = ""
        for word in answer.split():
            streamed += word + " "
            message_placeholder.write(streamed)
            time.sleep(0.02)

        # Show citations
        with st.expander("📚 Sources"):
            for i, src in enumerate(sources):
                st.write(f"**Chunk {i+1}:**")
                st.write(src)

    # Save conversation
    st.session_state.messages.append(
        ("user", query, None)
    )

    st.session_state.messages.append(
        ("assistant", answer, sources)
    )
