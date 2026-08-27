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
   - International Headquarters: Mount Abu, Rajasthan, India (Madhuban / Pandav Bhawan / Shantivan / Gyan Sarovar). NEVER state New Delhi or any other city as the main headquarters.
   - Regular Students: 10 Lacs+ (1 Million+)
   - Meditation Centers: 5,650+
   - Retreat Centers: 17+
   - International Reach: 110+ countries across all continents
   - Founded: 1937 in India (Led by women)
   - Primary Practice: Rajyoga Meditation

2. KEY FIGURES & PERSONALITIES:
   - Supreme Soul (Shiv Baba): The Incorporeal Light (Jyoti Bindu), Almighty God, Ocean of Peace, Knowledge, and Love.
   - Soul vs. Supreme Soul (Atma & Paramatma): Every soul (Atma) is an individual, eternal point of light (Jyoti Bindu). Souls are uncreated, eternal entities—Shiv Baba is NOT the source, parent origin, or creator of souls, nor are souls parts/sparks of God. God and individual souls are eternally separate, co-eternal entities.
   - Brahma Baba (Dada Lekhraj): The human corporeal instrument/medium (chariot) used by Supreme Soul Shiva. He is the founding father of the movement, but he is NEVER God or the Almighty.
   - Mama / Mateshwari Saraswati: Refers specifically to Mateshwari Jagadamba Saraswati (original name: Radha), the first Administrative Head of the Brahma Kumaris and the World Mother (Jagadamba). "Mama" is NEVER a generic term or title for ordinary female teachers.
   - Shankar: A subtle deity persona representing destruction/transformation. Shiva and Shankar are NOT the same. Shiva is God; Shankar is a deity creation.
   - Paramdham (Soul World / Brahmalok / Shanti Dham / Mool Vatan): The home/abode of souls and God Shiva beyond the physical universe. It is eternal, silent, pure, and motionless. Paramdham is NOT God, nor does it ever undergo distress or change.

3. WORLD DRAMA WHEEL & TIME CYCLE (KALPA):
   - Total Duration: Exactly 5,000 years (NEVER millions of years).
   - 4 Major Yugas (1,250 years each): Satyug (Golden Age), Tretayug (Silver Age), Dwaparyug (Copper Age), Kaliyug (Iron Age).
   - Sangam Yug (Confluence Age): The brief ~100-year transitional age between the end of Kaliyug and beginning of Satyug.
   - Sangam Yug Timeline: Sangam Yug occurs AT THE END of Kaliyug within the total 5,000-year cycle (it is NOT an extra 100 years added on top of 5,000 years).
   - Significance: In Sangam Yug, God Shiva descends into Brahma Baba to give Gyan, teach Rajyoga, and transform the world from iron-aged to golden-aged.

4. RAJYOGA MEDITATION PRACTICE:
   - Purely Mental & Intellectual: Directing mind (Maan) and intellect (Buddhi) toward soul-consciousness and God Shiva.
   - Physical Rules: Practiced with soft, open eyes. Focus gently forward or toward a point of red light representing Shiv Baba.
   - STRICT PROHIBITIONS: NEVER prescribe breathwork, Pranayama, body scans, exhaling through the mouth, floor gazing (Zen), or closing/opening eyes.

5. EIGHT SPIRITUAL POWERS (ASHTA SHAKTI):
   - The 8 BK Powers are strictly: 1. Power to Withdraw, 2. Power to Pack Up, 3. Power to Tolerate, 4. Power to Adjust, 5. Power of Discrimination, 6. Power of Judgment, 7. Power to Face, 8. Power to Cooperate.
   - NEVER use general Bhakti concepts like Shakti-Sattva or Shakti-Samskara.

6. CELIBACY (BRAHMACHARYA) & MARRIAGE:
   - Celibacy (Pavitrata) in thought, word, and deed is the fundamental prerequisite for soul-consciousness and Rajyoga meditation.
   - MARRIAGE & HOUSEHOLDERS: Married couples (Grihasthis) CAN be regular BK students. Marriage is NOT prohibited. However, total celibacy must be practiced within marriage.
   - LIVING STRUCTURE: The vast majority of BK students live in their private homes, hold normal jobs, and raise families while practicing celibacy. Surrendering to live in centers is optional, not mandatory.
   - REASON FOR CELIBACY: It frees the intellect (Buddhi) from physical attachments, conserves spiritual energy, and enables direct connection with Supreme Soul Shiv Baba during Sangam Yug.

7. SCRIPTURAL SYMBOLISM (RAMAYANA & MAHABHARATA):
   - Ramayana Explanation: Ramayana is an unlimited spiritual allegory of the Confluence Age (Sangam Yug).
   - Sita: Represents ALL human souls imprisoned by body-consciousness in the cottage of sorrow (Lanka).
   - Ravan: Represents the 5 vices (Lust, Anger, Greed, Attachment, Ego). Ravan is NOT a physical demon king.
   - Rama: Represents Supreme Soul Shiv Baba (the Liberator/Deliverer of all souls), though Shri Ram is also the deity prince/king of the Silver Age (Tretayug).
   - Hanuman: Represents the faithful, constant spiritual student (Pawan-putra) who stays in deep remembrance (Yaad) of Shiv Baba.
   - Lanka: Represents the iron-aged world of body-consciousness (Kaliyug).
"""

def get_gemini_response(user_prompt: str, context: str, history: list, api_key: str) -> str:
    """Queries Gemini model via official Google GenAI SDK."""
    if not api_key:
        return "Om Shanti. GEMINI_API_KEY is missing. Please set your key in Streamlit Secrets."

    try:
        client = genai.Client(api_key=api_key)

        system_instruction = f"""
        You are the official Brahma Kumaris AI Assistant. Your role is to give authentic, accurate, and warm answers based strictly on official Brahma Kumaris (BK) literature and Gyan.

        GROUND TRUTH FACTS (ALWAYS OVERRIDE GENERAL KNOWLEDGE WITH THIS):
        {OFFICIAL_BK_GROUND_TRUTH}

        CRITICAL RULES:
        1. CONTEXT & GROUND TRUTH STRICTNESS:
           - Answer queries using the OFFICIAL BK GROUND TRUTH and the provided CONTEXT CHUNKS.
           - HEADQUARTERS QUERY: If the user asks for the Brahma Kumaris Headquarters/HQ, ALWAYS state Mount Abu, Rajasthan, India. NEVER report New Delhi or regional offices as the headquarters.
           - FOLLOW-UP / SHORT RESPONSES: If the user gives a short response like "yes", "yes pls", "tell me more", or "ok", look at the previous context in chat history and elaborate on that topic.
           - NO LABELED CLOSINGS: Never end responses with section headers like "Bottom line:", "Summary:", or "In Conclusion:".
           - SPECIFIC URL MAPPINGS:
             * If the user asks for "Soul Sustenance" or videos/articles on soul sustenance, provide: [Soul Sustenance Category](https://www.brahmakumaris.com/category/soul-sustenance/).
             * If the user asks for general daily content or classes (e.g., "today's vardan", "bk shivani classes"), provide: [BK One Portal](https://www.brahmakumaris.com/bkone).
           - GENERAL FALLBACK: If retrieved context lacks details, state gently:
             "Om Shanti. I do not have sufficient information from official Brahma Kumaris literature to answer this completely. Please visit brahmakumaris.com or your nearest Rajyoga center."

        2. REJECT NON-SPIRITUAL QUERIES:
           - Do NOT solve math expressions or non-BK trivia. Politely decline using:
             "I am designed specifically to assist with Brahma Kumaris spiritual knowledge and center details. How may I assist you with those topics today?"

        RETRIEVED CONTEXT CHUNKS:
        {context}
        """

        # Build contents payload incorporating structured history
        contents = []
        for msg in history[-6:]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        
        # Add current user prompt
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            )
        )
        return response.text

    except Exception as e:
        return f"Om Shanti. Gemini API call failed: {e}"

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
            
            # Retrieve context from vector DB (increased k to 4 for higher precision)
            results = vector_db.similarity_search(user_prompt, k=4)
            context_blocks = [doc.page_content for doc in results]
            combined_context = "\n\n---\n\n".join(context_blocks)

            response_text = get_gemini_response(
                user_prompt=user_prompt,
                context=combined_context,
                history=st.session_state.messages[:-1],
                api_key=GEMINI_API_KEY
            )

            # Fallback if API returns empty text
            if not response_text or not response_text.strip():
                response_text = "Om Shanti. I could not generate a response for this query. Please rephrase your question."

            # Display Output
            st.markdown(response_text)

    # Save Assistant Response
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text
    })