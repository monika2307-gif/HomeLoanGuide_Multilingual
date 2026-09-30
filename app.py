import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from config import (
    CATEGORY_MAPPINGS,
    SAMPLE_QUESTIONS,
    DEFAULT_HF_MODEL,
    FAISS_INDEX_DIR,
    DOCUMENTS_DIR,
)
from rag_engine import (
    load_faiss_index,
    build_faiss_index,
    ask_loan_guide,
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="LoanGuide AI — Multilingual Home Loan Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished financial assistant theme
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .badge-container {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .badge {
        background-color: #e0e7ff;
        color: #3730a3;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        display: inline-block;
    }
    .badge-bank {
        background-color: #fef3c7;
        color: #92400e;
    }
    .badge-scheme {
        background-color: #dcfce7;
        color: #166534;
    }
    .source-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .stChatMessage {
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    with st.spinner("Loading Multilingual Vector Knowledge Base..."):
        st.session_state.vectorstore = load_faiss_index()

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.markdown("### 🏦 LoanGuide AI")
    st.caption("Multilingual Housing Finance & Regulatory Expert")

    st.divider()

    # 1. Hugging Face API Configuration
    st.markdown("#### 🔑 Hugging Face API Token")
    env_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    hf_token = st.text_input(
        "HF Hub Token",
        value=env_token,
        type="password",
        help="Enter your Hugging Face API token. You can get one for free at https://huggingface.co/settings/tokens",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxxx",
    )

    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        st.success("API Token is active", icon="✅")
    else:
        st.warning("Please provide a Hugging Face API token to ask questions.", icon="⚠️")

    st.caption(f"LLM: `{DEFAULT_HF_MODEL}`")

    st.divider()

    # 2. Bank / Category Filter
    st.markdown("#### 📁 Guideline Filter")
    category_options = list(CATEGORY_MAPPINGS.keys())
    selected_category = st.selectbox(
        "Focus on specific guidelines:",
        options=category_options,
        index=0,
        help="Filter retrieval to specific bank or regulatory documents, or search across all guidelines.",
    )

    st.divider()

    # 3. 1-Click Sample Questions
    st.markdown("#### 💡 Quick Test Questions")
    st.caption("Click to test questions in English, Hindi, or Marathi:")

    for idx, item in enumerate(SAMPLE_QUESTIONS):
        btn_label = f"[{item['lang']}] {item['label']}"
        if st.button(btn_label, key=f"sample_{idx}", use_container_width=True):
            st.session_state.pending_query = item["query"]

    st.divider()

    # 4. Vector Database & Maintenance
    st.markdown("#### ⚙️ Knowledge Base")
    if st.session_state.vectorstore is not None:
        total_vectors = st.session_state.vectorstore.index.ntotal
        st.info(f"**FAISS Index Loaded**\n\nVectors: `{total_vectors:,}`", icon="📊")
    else:
        st.error("No FAISS index found. Click Rebuild below to create one.", icon="❌")

    if st.button("🔄 Rebuild FAISS Index", use_container_width=True):
        try:
            with st.spinner("Extracting PDF documents and rebuilding embeddings..."):
                vs, stats = build_faiss_index(DOCUMENTS_DIR)
                st.session_state.vectorstore = vs
                st.success(
                    f"Index rebuilt! Loaded {stats['pages_loaded']} pages into {stats['total_vectors']:,} vectors.",
                    icon="🎉",
                )
                st.rerun()
        except Exception as e:
            st.error(f"Error rebuilding index: {e}")

    # 5. Clear Chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()


# =========================================================
# MAIN CONTENT HEADER
# =========================================================
st.markdown('<div class="main-title">LoanGuide AI — Multilingual Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Instant, cited guidance on Home Loans, KYC, RBI circulars, interest rates, and PMAY-U 2.0.</div>',
    unsafe_allow_html=True,
)

# Badges
st.markdown(
    """
    <div class="badge-container">
        <span class="badge">🌐 English</span>
        <span class="badge">🌐 हिन्दी (Hindi)</span>
        <span class="badge">🌐 मराठी (Marathi)</span>
        <span class="badge badge-bank">🏦 ICICI Bank</span>
        <span class="badge badge-bank">🏦 State Bank of India</span>
        <span class="badge badge-scheme">🏛️ RBI Guidelines</span>
        <span class="badge badge-scheme">🏠 PMAY-U 2.0</span>
        <span class="badge badge-scheme">⚖️ Banking Ombudsman</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# WELCOME / EMPTY STATE
# =========================================================
if not st.session_state.messages:
    st.info(
        """
        👋 **Welcome to LoanGuide AI!**
        
        You can ask any home loan or regulatory question in **English**, **Hindi (हिन्दी)**, or **Marathi (मराठी)**.
        
        *Example questions you can ask:*
        - *"What documents are required for a home loan?"*
        - *"स्वरोजगार करने वाले व्यक्ति को होम लोन के लिए कौन से दस्तावेज चाहिए?"*
        - *"होम लोनसाठी कोणती कागदपत्रे आवश्यक आहेत?"*
        - *"What are the RBI rules on foreclosure / prepayment penalties?"*
        - *"What are the eligibility criteria for PMAY-U 2.0?"*
        
        Every answer strictly quotes the source document and page number from the official bank and regulatory circulars.
        """,
        icon="ℹ️",
    )


# =========================================================
# CHAT HISTORY DISPLAY
# =========================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If assistant message has retrieved chunks, show collapsible audit viewer
        if message["role"] == "assistant" and "chunks" in message and message["chunks"]:
            with st.expander(f"📚 View Retrieved Sources ({len(message['chunks'])} citations)"):
                for i, chunk in enumerate(message["chunks"], start=1):
                    st.markdown(
                        f"""
                        **Passage {i}:** `📄 {chunk['source']}` &nbsp;|&nbsp; `Page: {chunk['page']}`
                        > {chunk['content']}
                        ---
                        """
                    )


# =========================================================
# PROCESS NEW QUERY
# =========================================================
user_input = st.chat_input("Ask a question in English, हिन्दी, or मराठी...")

# If sample button was pressed, override user_input
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    # 1. Check if vectorstore is loaded
    if st.session_state.vectorstore is None:
        st.error("FAISS index is not loaded. Please build the index from the sidebar first.", icon="❌")
        st.stop()

    # 2. Check if HF token is provided
    active_token = hf_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    if not active_token:
        st.warning("Please enter your Hugging Face API token in the sidebar to generate answers.", icon="⚠️")
        st.stop()

    # 3. Append User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 4. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting loan guidelines and official documents..."):
            try:
                answer, chunks = ask_loan_guide(
                    vectorstore=st.session_state.vectorstore,
                    question=user_input,
                    api_token=active_token,
                    category_filter=selected_category,
                )

                st.markdown(answer)

                if chunks:
                    with st.expander(f"📚 View Retrieved Sources ({len(chunks)} citations)"):
                        for i, chunk in enumerate(chunks, start=1):
                            st.markdown(
                                f"""
                                **Passage {i}:** `📄 {chunk['source']}` &nbsp;|&nbsp; `Page: {chunk['page']}`
                                > {chunk['content']}
                                ---
                                """
                            )

                # Save assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": chunks,
                })

            except Exception as e:
                error_msg = f"An error occurred while answering: {e}"
                st.error(error_msg, icon="🚨")
                st.caption("Tip: Ensure your Hugging Face token is valid and has read permissions.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}",
                    "chunks": [],
                })
