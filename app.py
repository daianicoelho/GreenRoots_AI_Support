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
    page_title="GreenRoots AI | Support",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- DESIGN SYSTEM (Inter · Fira Code · Forest Green / Dark Neutral) ---
st.markdown("""
    <style>
    /* ── TOKENS ────────────────────────────────────────────────────────
       Primary   #18181E  #505058  #888890  #C0C0C8
       Secondary #3A5010  #4A5218  #6A7035  #8A9045
       Neutral   #080808  #1E1E28  #383848  #585868  #909098  #B8B8C0  #D8D8E0
       Accent    #1A4820  #204830  #2A5828  #387858  #408865
    ── ──────────────────────────────────────────────────────────────── */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    /* GLOBAL */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #D8D8E0;
    }
    code, pre, .stCode, [data-testid="stCodeBlock"] {
        font-family: 'Fira Code', monospace !important;
    }

    /* APP BACKGROUND */
    .stApp {
        background-color: #18181E;
        color: #D8D8E0;
    }

    /* MAIN TITLE */
    .main-title {
        color: #8A9045;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700;
        font-size: 1.75rem;
        letter-spacing: -0.02em;
        text-align: center;
        padding: 8px 0;
        margin-top: -24px;
        margin-bottom: 24px;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #1E1E28 !important;
        border-right: 1px solid #383848 !important;
        transform: translateX(0) !important;
        min-width: 244px !important;
        visibility: visible !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin: 0 auto;
    }
    button[title="View fullscreen"],
    [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }

    /* SIDEBAR DIVIDER */
    [data-testid="stSidebar"] hr {
        border-color: #383848 !important;
    }

    /* NAV BUTTONS */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        background-color: #1E1E28 !important;
        color: #B8B8C0 !important;
        border: 1px solid #383848 !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 8px 16px !important;
        margin-bottom: 8px !important;
        text-align: left !important;
        transition: background-color 0.2s, border-color 0.2s, color 0.2s;
    }
    .stButton > button:hover {
        background-color: #2A5828 !important;
        border-color: #408865 !important;
        color: #D8D8E0 !important;
    }

    /* CHAT INPUT */
    [data-testid="stChatInput"] {
        background-color: #1E1E28 !important;
        border: 1px solid #383848 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #408865 !important;
        box-shadow: 0 0 0 2px rgba(64,136,101,0.2) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        background-color: transparent !important;
        color: #D8D8E0 !important;
        border: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #585868 !important;
    }

    /* CHAT BUBBLES — Assistant */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #1E1E28 !important;
        border: 1px solid #383848 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
    }

    /* CHAT BUBBLES — User */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #252532 !important;
        border: 1px solid #383848 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
    }

    /* SUGGESTION BUTTONS */
    .suggestion-btn > button {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        background-color: #1E1E28 !important;
        color: #8A9045 !important;
        border: 1px solid #383848 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin-bottom: 8px !important;
        width: 100% !important;
        text-align: left !important;
        transition: background-color 0.2s, border-color 0.2s, color 0.2s;
    }
    .suggestion-btn > button:hover {
        background-color: #204830 !important;
        border-color: #408865 !important;
        color: #D8D8E0 !important;
    }

    /* EXPANDERS (Plant Care Guide) */
    [data-testid="stExpander"] {
        background-color: #1E1E28 !important;
        border: 1px solid #383848 !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #B8B8C0 !important;
    }

    /* INFO / WARNING BOXES */
    .stAlert {
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* CAPTION / SECONDARY TEXT */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #585868 !important;
        font-size: 0.75rem !important;
    }

    /* HINT TEXT (empty state) */
    .hint-text {
        text-align: center;
        color: #585868;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        margin-top: 24px;
        margin-bottom: 16px;
    }

    /* CHAT INPUT FOOTER — match app background */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    .stChatFloatingInputContainer,
    .stChatFloatingInputContainer > div {
        background-color: #18181E !important;
        border-top: none !important;
        box-shadow: none !important;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 2. AI ENGINE INITIALIZATION
_DB_PATH = "./vector_db"
_MAX_INPUT_LENGTH = 500

@st.cache_resource
def init_bot():
    # Support both local .env and Streamlit Cloud secrets
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not configured. "
            "Add it to .env locally or to Streamlit Cloud secrets."
        )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=_DB_PATH,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0
    )

    template = """You are GreenRoots' Virtual Assistant, a botanical expert.
    Answer the user's question ONLY based on the provided context. 
    Be polite, helpful, and use plant emojis.
    If the answer is not in the context, ask them to contact support@greenroots.com.

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
_NAV_LABELS = {
    'chat':  '🏠 Home - Chat',
    'about': '📖 About GreenRoots',
    'guide': '🌵 Plant Care Guide',
}

with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)

    st.markdown("<h2 style='text-align: center; color: #8A9045; font-family: Inter, sans-serif; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.01em;'>GreenRoots</h2>", unsafe_allow_html=True)
    st.divider()

    for _page, _label in _NAV_LABELS.items():
        _is_active = st.session_state.page == _page
        _display = f"**{_label} ◀**" if _is_active else _label
        if st.button(_display, key=f"nav_{_page}", use_container_width=True):
            navigate_to(_page)
            st.rerun()

    st.divider()

    if st.session_state.page == 'chat' and st.session_state.get('messages'):
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.caption("v1.0 | Powered by Llama 3.1")

# 4. MAIN CONTENT ROUTING
if st.session_state.page == 'chat':
    st.markdown("<h1 class='main-title'>🌿 GreenRoots AI Support</h1>", unsafe_allow_html=True)

    if not os.path.exists(_DB_PATH):
        st.error("Vector Database not found! Please run 'python ingestion.py' first.")
    else:
        try:
            bot = init_bot()
        except Exception as e:
            st.error(f"Bot initialization failed: {e}")
            bot = None

    _SUGGESTIONS = [
        "🌱 How often should I water my Peace Lily?",
        "☀️ Which plants are best for low-light rooms?",
        "🚚 What is GreenRoots' shipping policy?",
    ]

    if os.path.exists(_DB_PATH) and 'bot' in dir() and bot is not None:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Empty state: show suggestions when no messages yet
        if not st.session_state.messages:
            st.markdown(
                "<p class='hint-text'>Not sure what to ask? Try one of these:</p>",
                unsafe_allow_html=True,
            )
            for _suggestion in _SUGGESTIONS:
                st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
                if st.button(_suggestion, key=f"sug_{_suggestion}"):
                    st.session_state["_pending_input"] = _suggestion
                st.markdown('</div>', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle suggestion button clicks
        _pending = st.session_state.pop("_pending_input", None)

        user_input = st.chat_input("Ask me anything about our plants...") or _pending

        if user_input:
            user_input = user_input.strip()
            if len(user_input) > _MAX_INPUT_LENGTH:
                st.warning(
                    f"Your message is too long ({len(user_input)} chars). "
                    f"Please limit to {_MAX_INPUT_LENGTH} characters."
                )
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    try:
                        response = st.write_stream(bot.stream(user_input))
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        print(f"[ERROR] Chain invocation failed: {e}")
                        st.error("Sorry, I couldn't process your request. Please try again.")

elif st.session_state.page == 'about':
    st.markdown("<h1 class='main-title'>About Us</h1>", unsafe_allow_html=True)
    st.write("""
    GreenRoots is a technology-driven botanical boutique. 
    Our mission is to bridge the gap between nature and modern living through 
    expert knowledge and high-quality indoor plants.
    """)
    if st.button("← Back to Chat"):
        navigate_to('chat')
        st.rerun()

elif st.session_state.page == 'guide':
    st.markdown("<h1 class='main-title'>🌵 Plant Care Guide</h1>", unsafe_allow_html=True)
    st.info("General golden rules for plant parents. Click each topic to expand.")

    with st.expander("💧 Watering", expanded=True):
        st.markdown("""
        - Check if the **top inch of soil is dry** before watering — stick your finger in.
        - Water **slowly and deeply** until it drains from the bottom.
        - **Overwatering** is the #1 cause of houseplant death; when in doubt, wait.
        - Use **room-temperature water** to avoid shocking the roots.
        """)

    with st.expander("☀️ Light"):
        st.markdown("""
        | Window Direction | Light Level | Good Plants |
        |---|---|---|
        | South | Bright direct | Cacti, Succulents |
        | East / West | Bright indirect | Pothos, Monsteras |
        | North | Low light | Snake Plant, ZZ Plant |

        - Rotate your plants **¼ turn weekly** for even growth.
        - Yellowing leaves often signal **too much direct sun**.
        """)

    with st.expander("🌫️ Humidity & Temperature"):
        st.markdown("""
        - Most tropical plants thrive at **50–60% humidity**.
        - Group plants together to create a **microclimate** with higher humidity.
        - Avoid placing plants near **AC vents or heaters**.
        - Ideal temperature range: **18–27 °C (65–80 °F)**.
        """)

    with st.expander("🌱 Soil & Fertilizing"):
        st.markdown("""
        - Use a **well-draining potting mix** — compacted soil suffocates roots.
        - Fertilize every **2–4 weeks** during spring and summer; stop in winter.
        - Flush the soil with plain water once a month to prevent **salt build-up**.
        """)

    st.divider()
    if st.button("← Back to Chat"):
        navigate_to('chat')
        st.rerun()