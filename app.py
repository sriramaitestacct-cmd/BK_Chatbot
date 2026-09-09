import zipfile
import os
import time
import hashlib
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types
from google.genai.errors import APIError
import build_db

# Automatically extract pre-built vector DB if zip exists
if not os.path.exists("./chroma_db_bk") and os.path.exists("chroma_db_bk.zip"):
    with zipfile.ZipFile("chroma_db_bk.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Page Configuration
st.set_page_config(
    page_title="Brahma Kumaris Assistant",
    page_icon="🕉️",
    layout="centered"
)

DB_DIR = "./chroma_db_bk"
CACHE_DIR = "./chroma_response_cache"  # Persistent collection for LLM responses

st.title("🕉️ Brahma Kumaris AI Assistant (Pilot Test)")

st.caption(
    "**Om Shanti.** This AI assistant provides informational answers based on official Brahma Kumaris literature. "
    "As an experimental AI tool, responses may occasionally contain inaccuracies. "
    "For authentic spiritual study, Daily Murli, and guidance, please visit [brahmakumaris.com](https://www.brahmakumaris.com) "
    "or your nearest Rajyoga Meditation Center."
)

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is missing! Please configure your API key in Streamlit Cloud Secrets.")

# Initialize Embeddings & Vector DBs
@st.cache_resource
def load_vector_dbs():
    if not os.path.exists(DB_DIR):
        with st.spinner("Initializing knowledge base database for the first time..."):
            build_db.build_full_clean_vector_db()
            
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # 1. Main Knowledge Base Client
    kb_client = chromadb.PersistentClient(path=DB_DIR)
    kb_vector_db = Chroma(client=kb_client, embedding_function=embeddings)
    
    # 2. Response Semantic Cache Client
    cache_client = chromadb.PersistentClient(path=CACHE_DIR)
    cache_vector_db = Chroma(client=cache_client, collection_name="qa_cache", embedding_function=embeddings)
    
    return kb_vector_db, cache_vector_db

try:
    vector_db, response_cache = load_vector_dbs()
except Exception as e:
    st.error(f"Error loading databases: {e}")
    st.stop()

OFFICIAL_BK_GROUND_TRUTH = """
OFFICIAL BRAHMA KUMARIS GROUND TRUTH (NEVER DEVIATE FROM THESE FACTS):

1. CORE STATISTICS & HEADQUARTERS:
   - International Headquarters: Mount Abu, Rajasthan, India (Madhuban / Pandav Bhawan / Shantivan / Gyan Sarovar). NEVER state New Delhi or any other city as main HQ.
   - Founded: 1937 in India (Led by women)
   - Primary Practice: Rajyoga Meditation

2. KEY FIGURES & PERSONALITIES:
   - Supreme Soul (Shiv Baba): The Incorporeal Light (Jyoti Bindu), Almighty God, Ocean of Peace, Knowledge, and Love.
   - Soul vs. Supreme Soul (Atma & Paramatma): Every soul (Atma) is an individual, eternal point of light. Souls are uncreated, eternal entities—Shiv Baba is NOT the source or parent creator of souls, nor are souls sparks of God. God and individual souls are eternally separate, co-eternal entities.
   - Brahma Baba (Dada Lekhraj): The human corporeal chariot used by Supreme Soul Shiva. Founding father, but NEVER God.
   - Shankar: A subtle deity persona representing destruction/transformation. Shiva and Shankar are NOT the same. Shiva is God; Shankar is a deity creation.
   - Paramdham: The home of souls and God Shiva beyond the physical universe. Eternal, silent, pure.

3. WORLD DRAMA WHEEL & TIME CYCLE (KALPA):
   - Total Duration: Exactly 5,000 years.
   - 4 Major Yugas (1,250 years each): Satyug, Tretayug, Dwaparyug, Kaliyug.
   - Sangam Yug (Confluence Age): The brief ~100-year transitional age at the END of Kaliyug within the 5,000-year cycle.

4. RAJYOGA MEDITATION PRACTICE:
   - Purely Mental & Intellectual: Practiced with soft, open eyes directed toward a point of red light representing Shiv Baba.
   - STRICT PROHIBITIONS: NEVER prescribe breathwork, Pranayama, body scans, exhaling through mouth, or floor gazing.

5. EIGHT SPIRITUAL POWERS (ASHTA SHAKTI):
   - 1. Withdraw, 2. Pack Up, 3. Tolerate, 4. Adjust, 5. Discrimination, 6. Judgment, 7. Face, 8. Cooperate.

6. CELIBACY (BRAHMACHARYA) & MARRIAGE:
   - Celibacy in thought, word, and deed is essential for Rajyoga.
   - Married couples (Grihasthis) CAN be regular BK students by practicing celibacy within marriage at home.

7. SCRIPTURAL SYMBOLISM:
   - Ramayana: Allegory of Confluence Age. Sita=Human souls, Ravan=5 vices, Rama=Shiv Baba, Lanka=Iron-aged world.
"""

@st.cache_data(ttl=86400, show_spinner=False)
def query_vector_db(query: str):
    """Fetches vector context chunks and caches results in RAM."""
    results = vector_db.similarity_search(query, k=4)
    return "\n\n---\n\n".join([doc.page_content for doc in results])

def check_semantic_cache(query: str, similarity_threshold=0.88):
    """Checks if a semantically similar query was already answered by Gemini."""
    try:
        results = response_cache.similarity_search_with_relevance_scores(query, k=1)
        if results and len(results) > 0:
            doc, score = results[0]
            if score >= similarity_threshold:
                return doc.page_content
    except Exception:
        pass
    return None

def save_to_semantic_cache(query: str, response: str):
    """Saves completed Gemini response to local vector database for future reuse."""
    try:
        doc_id = hashlib.md5(query.lower().strip().encode()).hexdigest()
        response_cache.add_texts(
            texts=[response],
            metadatas=[{"original_query": query}],
            ids=[doc_id]
        )
    except Exception:
        pass

def fetch_uncached_gemini_response(user_prompt: str, context: str, history: list, api_key: str) -> str:
    """Executes Gemini API call, aggregates full response, and caches it locally."""
    system_instruction = f"""
    You are the official Brahma Kumaris AI Assistant. Provide concise, warm, authentic answers strictly based on official BK literature and ground truth.

    GROUND TRUTH FACTS:
    {OFFICIAL_BK_GROUND_TRUTH}

    CRITICAL RULES:
    1. BREVITY & CONCISION: Keep your response direct and under 150-200 words total to minimize token overhead.
    2. COMPLETION GUARANTEE: Ensure every output ends naturally with a complete sentence.
    3. COMPARISONS: Use concise Markdown tables or brief bullet points for comparison queries. Keep text inside cells short.
    4. HEADQUARTERS QUERY: Always state Mount Abu, Rajasthan, India.
    5. NO LABELED CLOSINGS: Do not end responses with section headers like 'Summary:', 'Bottom Line:', or 'In Conclusion:'.
    6. SPECIFIC URL MAPPINGS:
       - "Soul Sustenance": [Soul Sustenance Category](https://www.brahmakumaris.com/category/soul-sustenance/)
       - Daily Content/Classes: [BK One Portal](https://www.brahmakumaris.com/bkone)
    7. FALLBACK: If retrieved context lacks details, state:
       "Om Shanti. I do not have sufficient information from official Brahma Kumaris literature to answer this completely. Please visit brahmakumaris.com or your nearest Rajyoga center."
    8. REJECT NON-SPIRITUAL QUERIES: Politely decline math or general non-BK trivia questions.

    RETRIEVED CONTEXT CHUNKS:
    {context}
    """

    contents = []
    for msg in history[-2:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 60000}
    )

    max_retries = 3
    full_text = ""
    for attempt in range(max_retries):
        try:
            response_stream = client.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    max_output_tokens=2048,
                )
            )

            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
            
            # Save response to local vector cache for zero-token reuse
            if full_text:
                save_to_semantic_cache(user_prompt, full_text)
            return full_text

        except APIError as e:
            err_str = str(e)
            if ("429" in err_str or "503" in err_str) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f"Om Shanti. Connection interrupted: {err_str}"
        except Exception as e:
            return f"Om Shanti. Request error: {str(e)}"

@st.cache_data(ttl=604800, show_spinner=False)
def get_cached_or_llm_response(user_prompt: str, context: str, history: list, api_key: str) -> str:
    """Exact Match Cache Wrapper: Checks Streamlit RAM cache before invoking LLM logic."""
    return fetch_uncached_gemini_response(user_prompt, context, history, api_key)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Om Shanti. How may I assist you with Brahma Kumaris knowledge today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # 1. CHECK SEMANTIC VECTOR CACHE FIRST (0 Tokens)
        cached_response = check_semantic_cache(user_prompt)
        
        if cached_response:
            st.markdown(cached_response)
            full_response = cached_response
        else:
            # 2. IF CACHE MISS, RUN LLM & SAVE RESPONSE
            try:
                combined_context = query_vector_db(user_prompt)
            except Exception:
                combined_context = ""

            with st.spinner("Refining response..."):
                full_response = get_cached_or_llm_response(
                    user_prompt=user_prompt,
                    context=combined_context,
                    history=st.session_state.messages[:-1],
                    api_key=GEMINI_API_KEY
                )
            st.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response
    })