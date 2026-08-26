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

# Streamlit App Disclaimer
st.caption(
    "**Om Shanti.** This AI assistant provides informational answers based on official Brahma Kumaris literature. "
    "As an experimental AI tool, responses may occasionally contain inaccuracies. "
    "For authentic spiritual study, Daily Murli, and guidance, please visit [brahmakumaris.com](https://www.brahmakumaris.com) "
    "or your nearest Rajyoga Meditation Center."
)

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

# SYSTEM KNOWLEDGE BASE & GROUND TRUTH FACT SHEET
OFFICIAL_BK_GROUND_TRUTH = """
OFFICIAL BRAHMA KUMARIS GROUND TRUTH (NEVER DEVIATE FROM THESE FACTS):

1. CORE STATISTICS:
   - Regular Students: 10 Lacs+ (1 Million+)
   - Meditation Centers: 5,650+
   - Retreat Centers: 17+
   - International Reach: 110+ countries across all continents
   - Founded: 1937 in India (Led by women)
   - Primary Practice: Rajyoga Meditation

2. WORLD DRAMA WHEEL & TIME CYCLE (KALPA):
   - Total Duration: Exactly 5,000 years (NEVER millions of years).
   - 4 Major Yugas (1,250 years each): Satyug (Golden Age), Tretayug (Silver Age), Dwaparyug (Copper Age), Kaliyug (Iron Age).
   - Sangam Yug (Confluence Age): The brief ~100-year transitional age between the end of Kaliyug and beginning of Satyug.
   - Sangam Yug Timeline: Sangam Yug occurs AT THE END of Kaliyug within the total 5,000-year cycle (it is NOT an extra 100 years added on top of 5,000 years).
   - Significance: In Sangam Yug, God Shiva descends into Brahma Baba to give Gyan, teach Rajyoga, and transform the world from iron-aged to golden-aged.

3. GOD SHIVA vs. BRAHMA BABA vs. SHANKAR:
   - Supreme Soul (Shiv Baba): The Incorporeal Light (Jyoti Bindu), Almighty God, Ocean of Peace, Knowledge, and Love.
   - Brahma Baba (Dada Lekhraj): The human corporeal instrument/medium (chariot) used by Supreme Soul Shiva. He is the founding father of the movement, but he is NEVER God or the Almighty.
   - Shankar: A subtle deity persona representing destruction/transformation. Shiva and Shankar are NOT the same. Shiva is God; Shankar is a deity creation.
   - Paramdham (Soul World / Brahmalok / Shanti Dham / Mool Vatan): The home/abode of souls and God Shiva beyond the physical universe. It is eternal, silent, pure, and motionless. Paramdham is NOT God, nor does it ever undergo distress or change.

4. RAJYOGA MEDITATION PRACTICE:
   - Purely Mental & Intellectual: Directing mind (Maan) and intellect (Buddhi) toward soul-consciousness and God Shiva.
   - Physical Rules: Practiced with soft, open eyes. Focus gently forward or toward a point of red light representing Shiv Baba.
   - STRICT PROHIBITIONS: NEVER prescribe breathwork, Pranayama, body scans, exhaling through the mouth, floor gazing (Zen), or closing/opening eyes.

5. EIGHT SPIRITUAL POWERS (ASHTA SHAKTI):
   - The 8 BK Powers are strictly: 1. Power to Withdraw, 2. Power to Pack Up, 3. Power to Tolerate, 4. Power to Adjust, 5. Power of Discrimination, 6. Power of Judgment, 7. Power to Face, 8. Power to Cooperate.
   - NEVER use general Bhakti concepts like Shakti-Sattva or Shakti-Samskara.
"""

def get_llm_response(chat_history: list, retrieved_context: str, api_key: str) -> str:
    """Queries Groq LLM with context, conversation history, and strict behavior rules."""
    client = Groq(api_key=api_key)
    
    system_prompt = f"""
    You are the official Brahma Kumaris AI Assistant. Your role is to give authentic, accurate, and warm answers based strictly on official Brahma Kumaris (BK) literature and Gyan.

    GROUND TRUTH FACTS (ALWAYS OVERRIDE GENERAL KNOWLEDGE WITH THIS):
    {OFFICIAL_BK_GROUND_TRUTH}

    CRITICAL RULES:
    1. CONTEXT & GROUND TRUTH STRICTNESS:
       - Answer queries using the OFFICIAL BK GROUND TRUTH and the provided CONTEXT CHUNKS.
       - NEVER default to general Hindu mythology, Puranic timelines (millions of years), or Bhakti definitions.
       - MEDIA & CLASSES REDIRECTION (MURLI / VARDAN / BLESSING / SLOGAN / CLASSES / TALKS / SPEAKERS):
         If the user asks for ANY spiritual classes, talks, lectures, audio/video streams, or daily content (including specific speakers like BK Shivani, Suraj Bhai, etc., e.g., "bk shivani classes", "today's vardan", "daily class", "rajyoga class audio"), inform them warmly that official classes, audio/video lectures, Murlis, and daily spiritual study material are hosted directly on the BK One Portal, and provide the exact link: [BK One Portal](https://www.brahmakumaris.com/bkone).
       - GENERAL FALLBACK: If the retrieved context and ground truth lack details for general non-daily topics, state gently:
         "Om Shanti. I do not have sufficient information from official Brahma Kumaris literature to answer this completely. Please visit brahmakumaris.com or your nearest Rajyoga center."
       
    2. REJECT NON-SPIRITUAL QUERIES:
       - Do NOT solve math expressions, code snippets, or general non-BK trivia. Politely decline using this exact standard message without follow-up questions:
         "I am designed specifically to assist with Brahma Kumaris spiritual knowledge and center details. How may I assist you with those topics today?"

    3. TERMINOLOGY GUARDRAILS:
       - SOUL vs. SOUL WORLD: Souls (living points of light) undergo rebirth, distress, and purification. The Soul World (Paramdham) is the eternal, silent home that never experiences distress or mood changes.
       - SANGAM YUG: Always include the Confluence Age (~100 years) when explaining the World Drama Wheel or Time Cycle, specifying that it occurs at the end of Kaliyug within the total 5,000-year cycle.

    4. LINKING RULES (STRICT):
       - MURLI, VARDAN, BLESSING, SLOGAN, CLASSES, TALKS & MEDIA: Direct users ONLY to [BK One Portal](https://www.brahmakumaris.com/bkone) for Murli, Vardan, blessings, slogans, classes (including BK Shivani classes), talks, daily audio, streams, and downloads.
       - FINDING PHYSICAL CENTERS: Direct users to the official [Center Finder](https://www.brahmakumaris.com/centers/) ONLY when they explicitly ask to locate a physical center, physical address, city location, or phone number.
       - Markdown Links: Use format [Link Text](URL) ONLY when referencing exact URLs present in context or explicit rules. Never fabricate links.

    5. OUTPUT & CLOSING STYLE:
       - Clean format: Do NOT output raw shortcodes, bracketed tags (e.g. [drts-directory-search]), or code blocks.
       - Conversational Closing: End valid spiritual answers (including class/media redirections) with a warm follow-up question inviting further spiritual discussion. NEVER append follow-up questions to rejection messages.

    CONTEXT CHUNKS:
    {retrieved_context}
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append up to last 4 messages for context
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Gracefully handle Rate Limits (HTTP 429) without crashing the Streamlit app
        if "rate_limit" in str(e).lower() or "429" in str(e):
            return "Om Shanti. High request volume detected. Please wait 1-2 minutes before sending another query."
        else:
            return f"Om Shanti. An unexpected error occurred: {e}"

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
        with st.spinner("Searching official BK knowledge base..."):
            
            # Reduced k from 5 to 3 to optimize prompt token limits and stay under rate limits
            results = vector_db.similarity_search(user_prompt, k=3)
            
            context_blocks = []
            for doc in results:
                context_blocks.append(doc.page_content)
            
            combined_context = "\n\n---\n\n".join(context_blocks)

            # Query Groq LLM with chat history
            response_text = get_llm_response(st.session_state.messages, combined_context, GROQ_API_KEY)

            # Display Output
            st.markdown(response_text)

    # Save Assistant Response
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text
    })