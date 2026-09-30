# 🏦 Multilingual LoanGuide AI

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=monika2307-gif/HomeLoanGuide_Multilingual&branch=main&mainModule=app.py)

An intelligent, multi-turn, multilingual Retrieval-Augmented Generation (RAG) assistant for Home Loans, Housing Finance, and Banking Regulations in India. Built with **LangChain**, **FAISS**, **Hugging Face**, and **Streamlit**.

Supports queries in:
- 🇬🇧 **English**
- 🇮🇳 **हिन्दी (Hindi)**
- 🇮🇳 **मराठी (Marathi)**

---

## 🌟 Key Features

1. **Exact Official Citations**: Every answer cites the exact PDF source name and 1-based page number (e.g. `(Source: ICICI_Home_Loan_Documents.pdf, Page: 2)` or `(स्रोत: ICICI_Home_Loan_Documents.pdf, पृष्ठ: 2)`).
2. **Anti-Hallucination Guardrails**: Strict prompt rules prevent inventing interest rates, eligibility criteria, or rules outside the provided context. If context is insufficient, a standardized multilingual fallback is provided.
3. **Multilingual Query Expansion**: Automatically enhances English, Hindi, and Marathi financial terms with domain-specific synonyms.
4. **Heuristic Document Routing & Re-ranking**: Dynamically routes queries to targeted circulars (ICICI, SBI, RBI, PMAY-U 2.0, Banking Ombudsman) and deduplicates chunks.
5. **Interactive Streamlit Web UI**:
   - 1-click test questions in all 3 languages.
   - Collapsible retrieved context expander for auditability.
   - Bank/Scheme category filter.
   - Sidebar FAISS index rebuilder and status monitor.

---

## 📂 Project Structure

```text
Loan_Guidelines(project)/
├── Documents/                        # Official PDF guidelines and circulars
│   ├── ICICI_Home_Loan_Documents.pdf
│   ├── ICICI_Home_Loan_FAQs.pdf
│   ├── SBI_HOME_LOANS_APPLICATION_FORM.pdf
│   ├── SBI_HOME_LOAN_MITC.pdf
│   ├── RBI_mastery.pdf
│   ├── RBI-Intresti-on-loans.pdf
│   ├── RBI-setlement.pdf
│   ├── RBi-penalty-charges.pdf
│   ├── RBI-Resposible-docs.pdf
│   ├── Ombudsman_Scheme_English.pdf
│   ├── Operational-Guidelines-of-PMAY-U-2.pdf
│   └── Operational-Guidelines-of-PMAY-U-2-Hindi.pdf
├── faiss_index/                      # Pre-computed FAISS vectorstore
│   ├── index.faiss
│   └── index.pkl
├── Notebook/                         # Original exploratory Jupyter notebook
│   └── Multilingual_Loan_GuideAI_Final (1).ipynb
├── app.py                            # Streamlit chat interface
├── config.py                         # Configuration, models, and sample questions
├── rag_engine.py                     # Retrieval, expansion, re-ranking & RAG chain
├── requirements.txt                  # Python dependencies
├── run_app.bat                       # 1-click Windows launcher
└── .env.example                      # Environment variables template
```

---

## 🚀 Quick Start

### 1. Requirements & Setup
Make sure you have Python installed with the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Hugging Face Token (Optional)
Create a `.env` file from `.env.example`:
```bash
copy .env.example .env
```
And add your token:
```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```
*(Alternatively, you can enter the token directly in the Streamlit sidebar at runtime).*

### 3. Run the Web Application
Double-click `run_app.bat` or run:
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.
