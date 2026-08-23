import os
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import build_db

# Page Configuration
st.set_page_config(
    page_title="Brahma Kumaris Assistant",
    page_icon="🕉️",
    layout="centered"
)

# Configuration Constants
DB_DIR = "./chroma_db_bk"
MODEL_NAME = "openai/gpt-oss-20b"

st.title("🕉️ Brahma Kumaris AI Assistant (Pilot Test)")

# Initialize Groq API Key
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

# Initialize Embeddings & Vector DB (Builds dynamically if missing on host)
@st.cache_resource
def load_vector_db():
    if not os.path.exists(DB_DIR):
        with st.spinner("Initializing knowledge base database for the first time..."):
            build_db.build_full_clean_vector_db()
            
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Initialize persistent client explicitly to prevent Rust binding locks
    client = chromadb.PersistentClient(path=DB_DIR)
    return Chroma(client=client, embedding_function=embeddings)

try:
    vector_db = load_vector_db()
except Exception as e:
    st.error(f"Error loading vector database: {e}")
    st.stop()

# SYSTEM KNOWLEDGE BASE
CORE_ORGANIZATION_FACTS = """
OFFICIAL BRAHMA KUMARIS CORE STATISTICS:
- Regular Students: 10 Lacs+ (1 Million+)
- Meditation Centers: 5,650+
- Retreat Centers: 17+
- International Reach: Spread over 110+ countries across all continents
- Founded: 1937 in India (Led by women)
- Primary Practice: Rajyoga Meditation
"""

def get_llm_response(chat_history: list, retrieved_context: str, api_key: str) -> str:
    """Queries Groq LLM with context, conversation history, and strict behavior rules."""
    client = Groq(api_key=api_key)
    
    system_prompt = f"""
    You are the official Brahma Kumaris AI Assistant.

    GUARANTEED CORE FACTS:
    {CORE_ORGANIZATION_FACTS}

    CRITICAL RULES:
    1. STRICT DOMAIN BOUNDARY: Answer ONLY questions related to Brahma Kumaris teachings, Rajyoga meditation, spiritual philosophy, centers, and courses.
    2. REJECT NON-SPIRITUAL OR OUT-OF-SCOPE QUERIES: Do NOT solve math expressions, code snippets, or general trivia. Politely decline using this exact standard message and DO NOT ask any follow-up questions:
       "I am designed specifically to assist with Brahma Kumaris spiritual knowledge and center details. How may I assist you with those topics today?"
    3. SPIRITUAL CONCEPT EQUIVALENCE: Recognize that Brahmalok, Shanti Dham, Paramdham, Nirvan Dham, and Mool Vatan all refer to the Incorporeal Soul World. Use provided context to explain these terms.
    4. BK ONE PORTAL ROUTING: For queries regarding Daily Murli, Avyakt Murlis, BK Audio, BK Tube, Books, Purusharth Charts, or internal student applications, provide a helpful general overview and direct the user to [BK One Portal](https://www.brahmakumaris.com/bkone).
    5. CONVERSATIONAL CLOSING: ONLY when giving a valid spiritual response, end with a warm follow-up question inviting further spiritual discussion. NEVER append a follow-up question if you are giving a fallback/rejection message.
    6. ACCURATE LINKING: Include markdown links [Link Text](URL) ONLY when referencing exact URLs explicitly present in the provided context chunks (or look for 'Page Source Link: <URL>'). Never invent or construct URLs.
    7. CLEAN OUTPUT: Do NOT output raw shortcodes, bracketed plugin tags (e.g. [drts-directory-search ...]), or code blocks.

    CONTEXT CHUNKS:
    {retrieved_context}
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append up to last 4 messages for context
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2
    )
    return completion.choices[0].message.content

def is_fallback_response(text: str) -> bool:
    """Detects standard guardrail or fallback statements in LLM response."""
    fallback_signatures = [
        "designed specifically to assist",
        "do not have information",
        "don't have information",
        "couldn't find relevant information",
        "how may i assist you with those topics"
    ]
    lower_text = text.lower()
    return any(sig in lower_text for sig in fallback_signatures)

# Chat Interface Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Om Shanti. How may I assist you with Brahma Kumaris knowledge today?", "sources": []}
    ]

# Display Existing Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📍 View Retrieved Source URLs"):
                for src in msg["sources"]:
                    st.markdown(f"- [{src}]({src})")

# User Query Processing
if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching official BK knowledge base..."):
            
            # Retrieve top 5 matching chunks from Vector Store
            results = vector_db.similarity_search(user_prompt, k=5)
            
            context_blocks = []
            sources = set()
            
            for doc in results:
                context_blocks.append(doc.page_content)
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
            
            combined_context = "\n\n---\n\n".join(context_blocks)

            # Query Groq LLM with chat history
            response_text = get_llm_response(st.session_state.messages, combined_context, GROQ_API_KEY)

            # Display Output
            st.markdown(response_text)
            
            # Only show sources if response is valid and NOT a fallback refusal
            valid_sources = []
            if sources and not is_fallback_response(response_text):
                valid_sources = list(sources)
                with st.expander("📍 View Retrieved Source URLs"):
                    for src in valid_sources:
                        st.markdown(f"- [{src}]({src})")

    # Save Assistant Response
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": valid_sources
    })