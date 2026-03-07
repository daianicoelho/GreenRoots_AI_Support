import streamlit as st
import os
from dotenv import load_dotenv

# AI Imports (LCEL Architecture)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. INITIAL CONFIGURATION
load_dotenv()
st.set_page_config(
    page_title="VerdeVida AI | Support", 
    page_icon="🌿", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- PREMIUM VISUAL DESIGN (MOSS GREEN & OBSIDIAN DARK) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    /* Main Title */
    .main-title {
        color: #8DAA71;
        font-family: 'Georgia', serif;
        font-weight: 700;
        text-align: center;
        padding: 10px;
        margin-top: -30px;
        margin-bottom: 20px;
    }

    /* SIDEBAR CUSTOMIZATION */
    [data-testid="stSidebar"] {
        background-color: #0A0D10 !important;
        border-right: 2px solid #35421C !important;
        min-width: 300px !important;
    }
    
    /* Sidebar Image Centering */
    [data-testid="stSidebar"] [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin-left: auto;
        margin-right: auto;
    }

    /* Disable Sidebar Expand Button */
    [data-testid="collapsedControl"], 
    button[title="View fullscreen"],
    [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }

    /* Sidebar Buttons Style */
    .stButton>button {
        background-color: #1A1E14 !important;
        color: #8DAA71 !important;
        border: 1px solid #35421C !important;
        width: 100% !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #35421C !important;
        color: #FFFFFF !important;
        border: 1px solid #8DAA71 !important;
    }

    /* CHAT INPUT CUSTOMIZATION */
    [data-testid="stChatInput"] {
        border: 1px solid #35421C !important;
        border-radius: 15px !important;
        background-color: #1A1E14 !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    /* CHAT BUBBLES */
    [data-testid="stChatMessage"] {
        background-color: #161B22 !important;
        border: 1px solid #24292F !important;
        border-radius: 15px;
        margin-bottom: 10px;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 2. AI ENGINE INITIALIZATION
@st.cache_resource
def init_bot():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_path = "./vector_db"
    
    if not os.path.exists(db_path):
        return None
        
    vectorstore = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    template = """You are VerdeVida's Virtual Assistant, a botanical expert.
    Answer the user's question ONLY based on the provided context. 
    Be polite, helpful, and use plant emojis.
    If the answer is not in the context, ask them to contact support@verdevida.com.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return chain

# 3. NAVIGATION LOGIC
if 'page' not in st.session_state:
    st.session_state.page = 'chat'

def navigate_to(page_name):
    st.session_state.page = page_name

# --- FIXED SIDEBAR (CLEAN VERSION) ---
with st.sidebar:
    # Centering the logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    
    st.markdown("<h2 style='text-align: center; color: #8DAA71;'>VerdeVida Menu</h2>", unsafe_allow_html=True)
    
    st.divider()
    
    if st.button("🏠 Home - Chat"):
        navigate_to('chat')
    
    if st.button("📖 About VerdeVida"):
        navigate_to('about')
    
    if st.button("🌵 Plant Care Guide"):
        navigate_to('guide')
        
    st.divider()
    
    st.caption("v1.0 | Powered by Llama 3.1")

# 4. MAIN CONTENT ROUTING
if st.session_state.page == 'chat':
    st.markdown("<h1 class='main-title'>🌿 VerdeVida AI Support</h1>", unsafe_allow_html=True)
    
    bot = init_bot()
    if bot is None:
        st.error("Vector Database not found! Please run 'python ingestion.py' first.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask me anything about our plants..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Consulting botanical database..."):
                    try:
                        response = bot.invoke(user_input)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Response Error: {e}")

elif st.session_state.page == 'about':
    st.markdown("<h1 class='main-title'>About Us</h1>", unsafe_allow_html=True)
    st.write("""
    VerdeVida is a technology-driven botanical boutique. 
    Our mission is to bridge the gap between nature and modern living through 
    expert knowledge and high-quality indoor plants.
    """)
    if st.button("← Back to Chat"):
        navigate_to('chat')
        st.rerun()

elif st.session_state.page == 'guide':
    st.markdown("<h1 class='main-title'>Plant Care Guide</h1>", unsafe_allow_html=True)
    st.info("General Golden Rules for Plant Parents:")
    st.write("- **Watering:** Always check if the top inch of soil is dry before watering.")
    st.write("- **Light:** Match your plant to your window's orientation.")
    st.write("- **Humidity:** Tropical plants love a good misting.")
    if st.button("← Back to Chat"):
        navigate_to('chat')
        st.rerun()