import zipfile
import os

# Automatically extract pre-built vector DB if zip exists
if not os.path.exists("./chroma_db_bk") and os.path.exists("chroma_db_bk.zip"):
    with zipfile.ZipFile("chroma_db_bk.zip", "r") as zip_ref:
        zip_ref.extractall(".")

import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types
import build_db

# Page Configuration
st.set_page_config(
    page_title="Brahma Kumaris Assistant",
    page_icon="🕉️",
    layout="centered"
)

# Configuration Constants
DB_DIR = "./chroma_db_bk"

st.title("🕉️ Brahma Kumaris AI Assistant (Pilot Test)")

# Streamlit App Disclaimer
st.caption(
    "**Om Shanti.** This AI assistant provides informational answers based on official Brahma Kumaris literature. "
    "As an experimental AI tool, responses may occasionally contain inaccuracies. "
    "For authentic spiritual study, Daily Murli, and guidance, please visit [brahmakumaris.com](https://www.brahmakumaris.com) "
    "or your nearest Rajyoga Meditation Center."
)

# Initialize Gemini API Key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is missing! Please configure your API key in Streamlit Cloud Secrets.")

# Initialize Embeddings & Vector DB
@st.cache_resource
def load_vector_db():
    if not os.path.exists(DB_DIR):
        with st.spinner("Initializing knowledge base database for the first time..."):
            build_db.build_full_clean_vector_db()
            
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    client = chromadb.PersistentClient(path=DB_DIR)
    return Chroma(client=client, embedding_function=embeddings)

try:
    vector_db = load_vector_db()
except Exception as e:
    st.error(f"Error loading vector database: {e}")
    st.stop()

# SYSTEM KNOWLEDGE BASE & GROUND TRUTH FACT SHEET
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

def get_gemini_stream(user_prompt: str, context: str, history: list, api_key: str):
    """Queries Gemini model in streaming mode safely without generator retry traps."""
    if not api_key:
        yield "Om Shanti. GEMINI_API_KEY is missing. Please set your key in Streamlit Secrets."
        return

    try:
        client = genai.Client(api_key=api_key)

        system_instruction = f"""
        You are the official Brahma Kumaris AI Assistant. Provide concise, warm, authentic answers strictly based on official BK literature and ground truth.

        GROUND TRUTH FACTS:
        {OFFICIAL_BK_GROUND_TRUTH}

        CRITICAL RULES:
        1. WORD COUNT & COMPLETION: Keep responses concise (under 200–250 words) and ensure your answer always concludes with a complete sentence.
        2. HEADQUARTERS QUERY: Always state Mount Abu, Rajasthan, India.
        3. NO LABELED CLOSINGS: Do not end responses with section headers like 'Summary:', 'Bottom Line:', or 'In Conclusion:'.
        4. SPECIFIC URL MAPPINGS:
           - "Soul Sustenance": [Soul Sustenance Category](https://www.brahmakumaris.com/category/soul-sustenance/)
           - Daily Content/Classes: [BK One Portal](https://www.brahmakumaris.com/bkone)
        5. FALLBACK: If retrieved context lacks details, state:
           "Om Shanti. I do not have sufficient information from official Brahma Kumaris literature to answer this completely. Please visit brahmakumaris.com or your nearest Rajyoga center."
        6. REJECT NON-SPIRITUAL QUERIES: Politely decline math or general non-BK trivia questions.

        RETRIEVED CONTEXT CHUNKS:
        {context}
        """

        # TOKEN OPTIMIZATION: Only pass the last 2 messages (1 turn) of chat history
        contents = []
        for msg in history[-2:]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        
        # Add current user prompt
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))

        # Streamed API Call
        response_stream = client.models.generate_content_stream(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                max_output_tokens=1024,
            )
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        err_str = str(e)
        if "429" in err_str:
            yield "\n\n[Om Shanti. High traffic detected. Please try asking again in a moment.]"
        elif "503" in err_str:
            yield "\n\n[Om Shanti. Service temporarily busy. Please retry.]"
        else:
            yield f"\n\n[Om Shanti. Connection interrupted: {err_str}]"

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
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # Retrieve context from vector DB (k=4 context chunks)
        results = vector_db.similarity_search(user_prompt, k=4)
        context_blocks = [doc.page_content for doc in results]
        combined_context = "\n\n---\n\n".join(context_blocks)

        # Stream response token-by-token using st.write_stream
        stream_generator = get_gemini_stream(
            user_prompt=user_prompt,
            context=combined_context,
            history=st.session_state.messages[:-1],
            api_key=GEMINI_API_KEY
        )
        
        full_response = st.write_stream(stream_generator)

    # Save Assistant Response
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response
    })