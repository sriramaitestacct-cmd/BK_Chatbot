import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

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
# Fetches key safely from Streamlit Cloud Secrets (or local environment)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

# Initialize Embeddings & Vector DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

try:
    vector_db = load_vector_db()
except Exception as e:
    st.error(f"Error loading vector database: {e}. Please ensure build_db.py has finished executing.")
    st.stop()

# ---------------------------------------------------------
# SYSTEM KNOWLEDGE BASE (Guaranteed Accurate Core Facts)
# ---------------------------------------------------------
CORE_ORGANIZATION_FACTS = """
OFFICIAL BRAHMA KUMARIS CORE STATISTICS:
- Regular Students: 10 Lacs+ (1 Million+)
- Meditation Centers: 5,650+
- Retreat Centers: 17+
- International Reach: Spread over 110+ countries across all continents
- Founded: 1937 in India (Led by women)
- Primary Practice: Rajyoga Meditation
"""

# ---------------------------------------------------------
# RESPONSE CACHING DECORATOR
# ---------------------------------------------------------
@st.cache_data(ttl=86400, max_entries=1000)
def get_llm_response(user_query: str, retrieved_context: str, api_key: str) -> str:
    """Queries Groq LLM with context. Cached for 24 hours to prevent duplicate API charges."""
    client = Groq(api_key=api_key)
    
    system_prompt = f"""
    You are the official Brahma Kumaris AI Assistant.
    Your sole task is to answer user queries accurately based strictly on the official Brahma Kumaris knowledge provided below.

    GUARANTEED CORE FACTS:
    {CORE_ORGANIZATION_FACTS}

    RULES:
    1. Prefer the GUARANTEED CORE FACTS above for core statistical queries (student count, center count, retreat centers, country count).
    2. Answer other questions strictly using the facts from the provided context chunks below.
    3. If the answer cannot be determined from either source, state clearly: "I am sorry, but I do not have official Brahma Kumaris information on this topic."
    4. Keep responses respectful, concise, spiritual, and aligned with Rajyoga philosophy.
    5. Include markdown links [Link Text](URL) when referencing specific pages.

    CONTEXT CHUNKS:
    {retrieved_context}
    """

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.2
    )
    return completion.choices[0].message.content


# Chat Interface Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Om Shanti. How may I assist you with Brahma Kumaris knowledge today?"}
    ]

# Display Existing Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Query Processing
if user_prompt := st.chat_input("Ask a question..."):
    # Display user input
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process response
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

            # Query Groq LLM using retrieved context
            response_text = get_llm_response(user_prompt, combined_context, GROQ_API_KEY)

            # Display Output & Sources
            st.markdown(response_text)
            
            if sources:
                with st.expander("📍 View Retrieved Source URLs"):
                    for src in sources:
                        st.markdown(f"- [{src}]({src})")

    # Store assistant response in session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})