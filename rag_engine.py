import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    DOCUMENTS_DIR,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_HF_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    CATEGORY_MAPPINGS,
)

# =========================================================
# EMBEDDING MODEL SINGLETON / FACTORY
# =========================================================
_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


# =========================================================
# FAISS INDEX LOADER & BUILDER
# =========================================================
def load_faiss_index() -> Optional[FAISS]:
    """Loads existing FAISS index from disk if available."""
    index_file = Path(FAISS_INDEX_DIR) / "index.faiss"
    if not index_file.exists():
        return None
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None


def build_faiss_index(documents_dir: Optional[Path] = None) -> Tuple[FAISS, Dict[str, Any]]:
    """Builds a new FAISS vector database from PDF documents."""
    doc_path = str(documents_dir or DOCUMENTS_DIR)
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Documents directory '{doc_path}' not found.")

    loader = PyPDFDirectoryLoader(doc_path)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No PDF documents found in '{doc_path}'.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Save to disk
    os.makedirs(str(FAISS_INDEX_DIR), exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    stats = {
        "pages_loaded": len(docs),
        "total_chunks": len(chunks),
        "total_vectors": vectorstore.index.ntotal,
    }
    return vectorstore, stats


# =========================================================
# MULTILINGUAL QUERY EXPANSION
# =========================================================
def expand_query(question: str) -> str:
    q = question.lower().strip()

    # PMAY-U 2.0
    if (
        "pmay" in q
        or "pmay-u" in q
        or "pmay u" in q
        or "pmay-u 2.0" in q
        or "pmay u 2.0" in q
        or "पीएमएवाई" in q
        or "प्रधानमंत्री आवास" in q
        or "आवास योजना" in q
    ):
        return (
            question
            + " PMAY-U 2.0 Pradhan Mantri Awas Yojana Urban "
              "housing scheme affordable housing "
              "beneficiary eligibility housing assistance "
              "Beneficiary Led Construction BLC "
              "Affordable Housing in Partnership AHP "
              "Interest Subsidy Scheme ISS"
        )

    # KYC
    elif (
        "kyc" in q
        or "know your customer" in q
        or "केवायसी" in q
        or "केवाईसी" in q
    ):
        return (
            question
            + " KYC Know Your Customer "
              "customer due diligence identity verification "
              "identity proof address proof PAN Aadhaar "
              "KYC update periodic KYC"
        )

    # INTEREST RATE
    elif (
        "interest" in q
        or "interest rate" in q
        or "व्याज" in q
        or "व्याजदर" in q
        or "ब्याज" in q
        or "ब्याज दर" in q
        or "floating rate" in q
    ):
        return (
            question
            + " home loan interest rate "
              "interest rate floating rate fixed rate "
              "benchmark rate lending rate "
              "loan interest calculation"
        )

    # EMI
    elif (
        "emi" in q
        or "equated monthly" in q
        or "ईएमआय" in q
        or "हप्ता" in q
        or "किस्त" in q
    ):
        return (
            question
            + " EMI Equated Monthly Instalment "
              "home loan EMI principal interest "
              "monthly instalment repayment "
              "loan tenure EMI calculation"
        )

    # REPAYMENT
    elif (
        "repayment" in q
        or "repay" in q
        or "परतफेड" in q
        or "परतफेडी" in q
        or "भुगतान" in q
        or "चुकौती" in q
    ):
        return (
            question
            + " home loan repayment "
              "loan repayment schedule EMI "
              "principal interest instalments "
              "repayment tenure"
        )

    # PREPAYMENT / FORECLOSURE
    elif (
        "prepayment" in q
        or "pre-payment" in q
        or "foreclosure" in q
        or "pre closure" in q
        or "pre-closure" in q
        or "फोरक्लोजर" in q
        or "पूर्व भुगतान" in q
        or "आगाऊ परतफेड" in q
    ):
        return (
            question
            + " home loan prepayment foreclosure "
              "pre-closure loan closure before maturity "
              "prepayment charges foreclosure charges "
              "outstanding principal loan settlement"
        )

    # GRIEVANCE / COMPLAINT / OMBUDSMAN
    elif (
        "complaint" in q
        or "grievance" in q
        or "ombudsman" in q
        or "complaint redressal" in q
        or "तक्रार" in q
        or "तक्रार निवारण" in q
        or "शिकायत" in q
        or "शिकायत निवारण" in q
        or "लोकपाल" in q
    ):
        return (
            question
            + " loan complaint grievance redressal "
              "bank complaint customer complaint "
              "RBI Integrated Ombudsman Scheme "
              "complaint escalation complaint resolution "
              "banking grievance"
        )

    # PROPERTY DOCUMENTS
    elif (
        "property document" in q
        or "property documents" in q
        or "मालमत्ता कागदपत्र" in q
        or "मालमत्तेची कागदपत्रे" in q
        or "संपत्ति दस्तावेज" in q
    ):
        return (
            question
            + " home loan property documents "
              "title deed property papers "
              "sale deed ownership proof "
              "encumbrance certificate property tax receipts"
        )

    # SALARIED APPLICANT
    elif (
        "salaried" in q
        or "salary" in q
        or "पगारदार" in q
        or "नोकरी" in q
    ):
        return (
            question
            + " salaried home loan applicant "
              "salary slips income proof bank statements "
              "Form 16 employment proof income documents"
        )

    # SELF-EMPLOYED
    elif (
        "self employed" in q
        or "self-employed" in q
        or "व्यावसायिक" in q
        or "स्वयंरोजगार" in q
    ):
        return (
            question
            + " self employed home loan applicant "
              "business proof income proof ITR "
              "bank statements financial statements "
              "business stability documents"
        )

    # HOME LOAN ELIGIBILITY
    elif (
        "eligibility" in q
        or "eligible" in q
        or "पात्रता" in q
        or "पात्र" in q
        or "eligibility criteria" in q
    ):
        return (
            question
            + " home loan eligibility criteria "
              "eligible borrower applicant "
              "income age repayment capacity "
              "credit history loan eligibility"
        )

    # HOME LOAN DOCUMENTS
    elif (
        "document" in q
        or "documents" in q
        or "कागदपत्र" in q
        or "कागदपत्रे" in q
        or "दस्तावेज" in q
        or "दस्तऐवज" in q
        or "कागजात" in q
    ):
        return (
            question
            + " home loan documents "
              "documents required for home loan "
              "identity proof address proof "
              "income proof salary slips bank statements "
              "property documents KYC documents"
        )

    return question


# =========================================================
# DETECT RELEVANT SOURCE DOCUMENTS
# =========================================================
def get_loan_sources(question: str) -> Optional[List[str]]:
    q = question.lower().strip()

    if (
        "pmay" in q
        or "pmay-u" in q
        or "pmay u" in q
        or "pmay-u 2.0" in q
        or "pmay u 2.0" in q
        or "पीएमएवाई" in q
        or "प्रधानमंत्री आवास" in q
        or "आवास योजना" in q
    ):
        return [
            "Operational-Guidelines-of-PMAY-U-2.pdf",
            "Operational-Guidelines-of-PMAY-U-2-Hindi.pdf",
        ]

    if "kyc" in q or "know your customer" in q or "केवायसी" in q or "केवाईसी" in q:
        return ["RBI_mastery.pdf"]

    if (
        "interest" in q
        or "interest rate" in q
        or "floating rate" in q
        or "emi" in q
        or "repayment" in q
        or "व्याज" in q
        or "व्याजदर" in q
        or "ब्याज" in q
        or "ईएमआय" in q
        or "परतफेड" in q
        or "किस्त" in q
    ):
        return ["RBI-Intresti-on-loans.pdf"]

    if (
        "prepayment" in q
        or "pre-payment" in q
        or "foreclosure" in q
        or "pre closure" in q
        or "pre-closure" in q
        or "पूर्व भुगतान" in q
        or "फोरक्लोजर" in q
        or "आगाऊ परतफेड" in q
    ):
        return ["RBI-setlement.pdf"]

    if (
        "penal charge" in q
        or "penalty" in q
        or "penal charges" in q
        or "दंड" in q
        or "दंडात्मक शुल्क" in q
    ):
        return ["RBi-penalty-charges.pdf"]

    if (
        "property document" in q
        or "property documents" in q
        or "release of property" in q
        or "title document" in q
        or "मालमत्ता कागदपत्र" in q
        or "मालमत्तेची कागदपत्रे" in q
    ):
        return ["RBI-Resposible-docs.pdf"]

    if (
        "complaint" in q
        or "grievance" in q
        or "ombudsman" in q
        or "complaint redressal" in q
        or "तक्रार" in q
        or "तक्रार निवारण" in q
        or "शिकायत" in q
        or "शिकायत निवारण" in q
        or "लोकपाल" in q
    ):
        return ["Ombudsman_Scheme_English.pdf"]

    if (
        "document" in q
        or "documents" in q
        or "कागदपत्र" in q
        or "कागदपत्रे" in q
        or "दस्तावेज" in q
        or "दस्तऐवज" in q
        or "कागजात" in q
    ):
        return [
            "ICICI_Home_Loan_Documents.pdf",
            "ICICI_Home_Loan_FAQs.pdf",
            "SBI_HOME_LOANS_APPLICATION_FORM.pdf",
            "SBI_HOME_LOAN_MITC.pdf",
        ]

    if "sbi" in q or "state bank" in q:
        return [
            "SBI_HOME_LOANS_APPLICATION_FORM.pdf",
            "SBI_HOME_LOAN_MITC.pdf",
        ]

    if (
        "home loan" in q
        or "housing loan" in q
        or "होम लोन" in q
        or "गृह कर्ज" in q
        or "गृह ऋण" in q
    ):
        return [
            "ICICI_Home_Loan_Documents.pdf",
            "ICICI_Home_Loan_FAQs.pdf",
            "SBI_HOME_LOANS_APPLICATION_FORM.pdf",
            "SBI_HOME_LOAN_MITC.pdf",
        ]

    return None


# =========================================================
# RERANKING & DEDUPLICATION LOGIC
# =========================================================
def rerank_loan_results(results: List[Tuple[Any, float]], query: str, top_k: int = 4):
    query_lower = query.lower()
    candidates = []

    keyword_groups = {
        "documents": [
            "document", "documents", "proof", "papers",
            "कागदपत्र", "दस्तऐवज", "दस्तावेज", "कागजात",
        ],
        "eligibility": [
            "eligible", "eligibility", "qualify",
            "पात्र", "पात्रता", "योग्य", "योग्यता",
        ],
        "kyc": ["kyc", "know your customer", "केवायसी", "केवाईसी"],
        "interest": [
            "interest", "interest rate", "floating rate",
            "व्याज", "व्याजदर", "ब्याज", "ब्याज दर",
        ],
        "emi": [
            "emi", "equated monthly instalment", "equated monthly installment",
            "ईएमआय", "हप्ता", "किस्त",
        ],
        "repayment": [
            "repayment", "repay", "payment",
            "परतफेड", "परतफेडी", "भुगतान", "चुकौती",
        ],
        "prepayment": [
            "prepayment", "foreclosure", "pre-close",
            "pre closure", "pre-closure", "आगाऊ परतफेड",
            "फोरक्लोजर", "पूर्व भुगतान",
        ],
        "pmay": [
            "pmay", "pmay-u", "pmay-u 2.0",
            "प्रधानमंत्री आवास योजना", "आवास योजना",
        ],
        "grievance": [
            "complaint", "grievance", "ombudsman",
            "complaint redressal", "तक्रार", "तक्रार निवारण",
            "शिकायत", "शिकायत निवारण", "लोकपाल",
        ],
        "loan": [
            "home loan", "housing loan", "loan",
            "होम लोन", "गृह कर्ज", "गृह ऋण",
        ],
    }

    query_keywords = []
    for group, keywords in keyword_groups.items():
        for keyword in keywords:
            if keyword in query_lower:
                query_keywords.append(group)
                break

    for doc, score in results:
        content = doc.page_content.lower()
        boost = 0.0

        matched_groups = 0
        for group in query_keywords:
            for keyword in keyword_groups[group]:
                if keyword in content:
                    matched_groups += 1
                    break

        boost -= matched_groups * 0.20

        if any(x in content for x in [
            "home loan documents",
            "documents required for home loan",
            "identity proof",
            "address proof",
            "income proof",
        ]):
            if "documents" in query_keywords:
                boost -= 0.50

        if any(x in content for x in [
            "eligibility criteria",
            "eligible borrower",
            "eligibility",
        ]):
            if "eligibility" in query_keywords:
                boost -= 0.50

        if any(x in content for x in [
            "know your customer",
            "kyc",
            "customer due diligence",
        ]):
            if "kyc" in query_keywords:
                boost -= 0.50

        if any(x in content for x in [
            "interest rate",
            "floating rate",
            "repo linked",
            "interest rates",
        ]):
            if "interest" in query_keywords:
                boost -= 0.50

        if any(x in content for x in [
            "equated monthly instalment",
            "equated monthly installment",
            "emi",
            "repayment",
        ]):
            if any(x in query_keywords for x in ["emi", "repayment"]):
                boost -= 0.50

        if any(x in content for x in [
            "prepayment",
            "foreclosure",
            "pre-closure",
            "pre closure",
        ]):
            if "prepayment" in query_keywords:
                boost -= 0.60

        if any(x in content for x in [
            "pmay-u",
            "pmay-u 2.0",
            "pradhan mantri awas yojana",
            "interest subsidy scheme",
            "beneficiary led construction",
        ]):
            if "pmay" in query_keywords:
                boost -= 0.70
            else:
                boost += 0.15

        if any(x in content for x in [
            "ombudsman",
            "grievance",
            "complaint",
            "complaint redressal",
        ]):
            if "grievance" in query_keywords:
                boost -= 0.70

        if "pmay" not in query_keywords:
            pmay_signals = [
                "pmay-u",
                "pmay-u 2.0",
                "pradhan mantri awas yojana",
                "beneficiary led construction",
            ]
            pmay_count = sum(content.count(s) for s in pmay_signals)
            if pmay_count >= 2:
                boost += 0.30

        final_score = score + boost
        candidates.append((doc, final_score, score))

    candidates.sort(key=lambda x: x[1])

    unique = []
    seen = set()
    for doc, final_score, original_score in candidates:
        text_key = re.sub(r"\s+", " ", doc.page_content.strip().lower())
        if text_key not in seen:
            seen.add(text_key)
            unique.append((doc, final_score, original_score))
        if len(unique) >= top_k:
            break

    return unique


# =========================================================
# RETRIEVE DOCUMENTS PIPELINE
# =========================================================
def retrieve_documents(
    vectorstore: FAISS,
    question: str,
    category_filter: Optional[str] = None,
    top_k: int = 4,
) -> List[Any]:
    expanded_question = expand_query(question)

    # Check explicit manual category filter first
    allowed_sources = None
    if category_filter and category_filter in CATEGORY_MAPPINGS:
        allowed_sources = CATEGORY_MAPPINGS[category_filter]

    # If no manual category filter or it's "All Documents", check question heuristic
    if not allowed_sources:
        allowed_sources = get_loan_sources(question)

    results = vectorstore.similarity_search_with_score(
        expanded_question,
        k=100,
    )

    if allowed_sources:
        filtered_results = []
        for doc, score in results:
            source = os.path.basename(doc.metadata.get("source", ""))
            if source in allowed_sources:
                filtered_results.append((doc, score))

        if filtered_results:
            results = filtered_results

    reranked_results = rerank_loan_results(
        results,
        question,
        top_k=top_k,
    )

    final_documents = [doc for doc, _, _ in reranked_results]
    return final_documents


# =========================================================
# FORMAT RETRIEVED DOCUMENTS FOR LLM
# =========================================================
def format_docs(docs: List[Any]) -> str:
    formatted_docs = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "Unknown")
        if isinstance(page, int):
            page = page + 1
        formatted_docs.append(
            f"[Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_docs)


# =========================================================
# SYSTEM PROMPT TEMPLATE
# =========================================================
PROMPT_TEMPLATE = """You are LoanGuide AI, a multilingual RAG-based Home Loan and
Housing Finance information assistant.

Your job is to answer the user's question using ONLY the information
provided in the CONTEXT.

============================================================
IMPORTANT RULES
============================================================

1. Use ONLY the information available in the CONTEXT.

2. Do NOT use outside knowledge, memory, assumptions, or general
knowledge.

3. Do NOT invent loan rules, eligibility criteria, charges, interest
rates, documents, government benefits, or procedures.

4. If the answer is clearly available in the CONTEXT, answer the
question directly.

5. Use the most directly relevant passage from the CONTEXT.

6. Keep the answer focused on exactly what the user asked.

7. Do NOT add unnecessary information that is not required to answer
the question.

8. If multiple documents contain relevant information, use the
information that directly answers the question and cite the relevant
sources.

9. If the question is about a specific bank, use information from that
bank's documents when available.

10. Do NOT assume that a rule from one bank applies to another bank.

11. For RBI rules or regulatory information, rely only on the RBI
documents present in the CONTEXT.

12. For PMAY-U 2.0 questions, rely only on the PMAY-U 2.0 documents
present in the CONTEXT.

13. Do NOT mix information from unrelated documents just because the
words are similar.

============================================================
LANGUAGE RULES
============================================================

Answer in the SAME LANGUAGE as the user's question.

English question → English answer.

Marathi question → Marathi answer.

Hindi question → Hindi answer.

Mixed Marathi/English → simple Marathi.

Mixed Hindi/English → simple Hindi.

Use simple and easy-to-understand language.

============================================================
SOURCE AND PAGE CITATION RULES
============================================================

Every factual answer MUST include the source document and page number.

Use ONLY the source name and page number explicitly provided in the
CONTEXT.

NEVER invent a source.

NEVER invent a page number.

NEVER use a page number from memory.

NEVER cite a page that is not present in the CONTEXT.

If the CONTEXT contains:

[Source: ICICI_Home_Loan_Documents.pdf, Page: 2]

then cite exactly:

(Source: ICICI_Home_Loan_Documents.pdf, Page: 2)

For Marathi:

(स्रोत: ICICI_Home_Loan_Documents.pdf, पृष्ठ: 2)

For Hindi:

(स्रोत: ICICI_Home_Loan_Documents.pdf, पेज: 2)

============================================================
MULTIPLE SOURCES
============================================================

If the answer uses information from multiple documents, cite each
relevant source.

Example:

(Source: RBI_mastery.pdf, Page: 15)
(Source: SBI_HOME_LOAN_MITC.pdf, Page: 3)

Do NOT cite documents that were not used to answer the question.

============================================================
BANK-SPECIFIC INFORMATION
============================================================

If the user asks:

"ICICI Bank home loan documents"

answer using ICICI-related context.

If the user asks:

"SBI home loan documents"

answer using SBI-related context.

Do NOT combine ICICI and SBI requirements unless the user explicitly
asks for a comparison.

============================================================
GOVERNMENT SCHEME INFORMATION
============================================================

For PMAY-U 2.0 questions:

Use only the PMAY-U 2.0 information available in the CONTEXT.

Do not use unrelated home-loan information unless it directly answers
the question.

============================================================
KYC / RBI INFORMATION
============================================================

For KYC questions:

Use only the RBI/KYC information available in the CONTEXT.

Do not assume that a general KYC rule applies to every bank unless the
CONTEXT supports that statement.

============================================================
PREPAYMENT / FORECLOSURE
============================================================

For prepayment or foreclosure questions:

Answer only from the retrieved documents.

Do not invent or assume charges, exemptions, timelines, or conditions.

If the context contains bank-specific or loan-type-specific conditions,
clearly mention that they are specific to that source.

============================================================
FALLBACK RULE
============================================================

Use the following fallback ONLY when the CONTEXT genuinely does not
contain enough information to answer the question:

"I do not have enough information in the provided documents to answer
this question. Please consult the relevant bank, RBI, government
department, or authorized person for more clarity."

Marathi:

"दिलेल्या कागदपत्रांमध्ये या प्रश्नाचे उत्तर देण्यासाठी पुरेशी
माहिती उपलब्ध नाही. अधिक माहितीसाठी संबंधित बँक, RBI, सरकारी विभाग
किंवा अधिकृत व्यक्तीशी संपर्क साधा."

Hindi:

"दिए गए दस्तावेज़ों में इस प्रश्न का उत्तर देने के लिए पर्याप्त
जानकारी उपलब्ध नहीं है। अधिक जानकारी के लिए संबंधित बैंक, RBI,
सरकारी विभाग या अधिकृत व्यक्ति से संपर्क करें।"

IMPORTANT:

If the CONTEXT contains enough information to answer the question,
DO NOT use the fallback.

NEVER give an answer followed by the fallback.

The response must be either:

A) A complete answer with citation

OR

B) The fallback response.

Never both.

============================================================
CONTEXT
============================================================

{context}

============================================================
QUESTION
============================================================

{question}

============================================================
FINAL CHECK
============================================================

Before answering, verify:

- Did I answer only from the CONTEXT?
- Did I answer the actual question?
- Did I use the correct language?
- Did I use simple language?
- Did I use the exact source name?
- Did I use the exact page number shown in the CONTEXT?
- Did I avoid inventing information?
- Did I avoid mixing unrelated bank information?
- Did I avoid unnecessary details?
- Did I avoid giving a fallback when the answer is available?
- Did I avoid citing unused documents?

Now provide the final answer.
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# =========================================================
# CHAT MODEL & RAG PIPELINE
# =========================================================
def get_chat_model(
    api_token: Optional[str] = None,
    repo_id: str = DEFAULT_HF_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> ChatHuggingFace:
    token = api_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    if not token:
        try:
            import streamlit as st
            if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
                token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        except Exception:
            pass

    if not token:
        raise ValueError(
            "Hugging Face API Token is required. Please set HUGGINGFACEHUB_API_TOKEN "
            "in your .env file, Streamlit secrets, or enter it in the Streamlit sidebar."
        )

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=llm)


def ask_loan_guide(
    vectorstore: FAISS,
    question: str,
    api_token: Optional[str] = None,
    repo_id: str = DEFAULT_HF_MODEL,
    category_filter: Optional[str] = None,
    top_k: int = 4,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executes retrieval and generation.
    Returns: (answer_string, list_of_retrieved_chunk_info)
    """
    retrieved_docs = retrieve_documents(
        vectorstore=vectorstore,
        question=question,
        category_filter=category_filter,
        top_k=top_k,
    )

    formatted_context = format_docs(retrieved_docs)
    chat_model = get_chat_model(api_token=api_token, repo_id=repo_id)

    rag_chain = prompt | chat_model | StrOutputParser()

    answer = rag_chain.invoke({
        "context": formatted_context,
        "question": question,
    })

    # Prepare metadata list for UI inspection
    chunks_meta = []
    for doc in retrieved_docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", 0)
        if isinstance(page, int):
            page = page + 1
        chunks_meta.append({
            "source": source,
            "page": page,
            "content": doc.page_content.strip(),
        })

    return answer, chunks_meta
