import streamlit as st
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Brahma Kumaris AI Assistant", page_icon="🕉️")
st.title("🕉️ Brahma Kumaris AI Assistant")
st.caption("Answers powered directly from www.brahmakumaris.com")

with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    st.markdown("[Get a free Groq API key here](https://console.groq.com/keys)")

if not groq_api_key:
    st.info("🔑 Please enter your Groq API Key in the sidebar to test the chatbot.")
    st.stop()

client = Groq(api_key=groq_api_key)

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db_bk", embedding_function=embeddings)

try:
    vector_db = load_vector_db()
except Exception as e:
    st.error("Vector DB not found! Please run 'python build_db.py' in Command Prompt first.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about BK knowledge..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    docs = vector_db.similarity_search(prompt, k=4)
    context_text = "\n\n".join([f"Source ({doc.metadata.get('source')}):\n{doc.page_content}" for doc in docs])

    system_instruction = f"""
    You are the official Brahma Kumaris AI Assistant.
    Your task is to answer user queries STRICTLY using only the provided retrieved context below.

    STRICT BOUNDARY RULES:
    1. Base your answer ONLY on the explicit facts provided in the 'Retrieved Web Context'. Do NOT use outside general knowledge or pre-trained memory.
    2. If the user's question cannot be answered using ONLY the provided context, respond with: 
       "I do not have information about this in my official Brahma Kumaris knowledge base."
    3. Synthesize the provided context naturally. Do NOT repeat raw UI elements or button text literally.
    4. If clickable links (e.g., [Text](URL)) are in the context, preserve them in your answer.

    Retrieved Web Context:
    {context_text}
    """

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Updated model parameter to an active Groq model ID
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        response_text = completion.choices[0].message.content
        message_placeholder.markdown(response_text)
        
        # Display retrieved sources for verification
        with st.expander("📚 View Retrieved Sources"):
            for doc in docs:
                st.write(f"- [{doc.metadata.get('source')}]({doc.metadata.get('source')})")

    st.session_state.messages.append({"role": "assistant", "content": response_text})