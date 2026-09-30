import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "Documents"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"

# Text Splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Defaults
DEFAULT_HF_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2

# Category / Bank Filters mapping to PDF filenames
CATEGORY_MAPPINGS = {
    "All Documents": None,
    "ICICI Bank": [
        "ICICI_Home_Loan_Documents.pdf",
        "ICICI_Home_Loan_FAQs.pdf",
    ],
    "State Bank of India (SBI)": [
        "SBI_HOME_LOANS_APPLICATION_FORM.pdf",
        "SBI_HOME_LOAN_MITC.pdf",
    ],
    "RBI Guidelines & Regulations": [
        "RBI_mastery.pdf",
        "RBI-Intresti-on-loans.pdf",
        "RBI-setlement.pdf",
        "RBi-penalty-charges.pdf",
        "RBI-Resposible-docs.pdf",
        "RBI-charges.pdf",
        "RBI-keywords.pdf",
        "Ombudsman_Scheme_English.pdf",
    ],
    "PMAY-U 2.0 (Housing Scheme)": [
        "Operational-Guidelines-of-PMAY-U-2.pdf",
        "Operational-Guidelines-of-PMAY-U-2-Hindi.pdf",
        "Operational-Guidelines-of-PMAY-U.pdf",
    ],
}

# Example Quick Queries for Streamlit
SAMPLE_QUESTIONS = [
    {
        "label": "Home Loan Documents (English)",
        "query": "What documents are required for a home loan?",
        "lang": "English",
    },
    {
        "label": "होम लोन के दस्तावेज (Hindi)",
        "query": "होम लोन के लिए कौन से दस्तावेज आवश्यक हैं?",
        "lang": "Hindi",
    },
    {
        "label": "होम लोन कागदपत्रे (Marathi)",
        "query": "होम लोनसाठी कोणती कागदपत्रे आवश्यक आहेत?",
        "lang": "Marathi",
    },
    {
        "label": "PMAY-U 2.0 Guidelines (English)",
        "query": "What is PMAY-U 2.0 and who is eligible?",
        "lang": "English",
    },
    {
        "label": "स्वरोजगार व्यक्ती होम लोन (Hindi)",
        "query": "स्वरोजगार करने वाले व्यक्ति को होम लोन के लिए कौन से दस्तावेज चाहिए?",
        "lang": "Hindi",
    },
    {
        "label": "RBI Prepayment Charges (English)",
        "query": "What are the RBI rules on prepayment charges for floating rate home loans?",
        "lang": "English",
    },
    {
        "label": "बँक तक्रार निवारण / लोकपाल (Marathi)",
        "query": "बँकेने तक्रार निवारण न केल्यास लोकपालकडे कशी तक्रार करावी?",
        "lang": "Marathi",
    },
]
