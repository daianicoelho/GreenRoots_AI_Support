import os
from dotenv import load_dotenv

# Modern LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. INITIAL SETUP
load_dotenv()

def run_bot():
    # Path to your vector database
    db_path = "./vector_db"
    
    if not os.path.exists(db_path):
        print("Error: Vector database not found. Please run 'python ingestion.py' first.")
        return

    # 2. INITIALIZE COMPONENTS
    print("Initializing VerdeVida AI Engine...")
    
    # Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load Vector Store
    vectorstore = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize Groq LLM (Llama 3.1)
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # 3. PROMPT & CHAIN SETUP (LCEL)
    template = """You are VerdeVida's Virtual Assistant, a botanical expert.
    Answer the user's question ONLY based on the provided context. 
    Be polite, helpful, and use plant emojis.
    If the answer is not in the context, ask them to contact support@verdevida.com.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Modular LCEL Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # 4. INTERACTIVE LOOP
    print("\n🌿 VerdeVida AI Support is Ready!")
    print("(Type 'exit' or 'quit' to stop the bot)\n")

    while True:
        query = input("👤 You: ")
        
        if query.lower() in ["exit", "quit", "bye"]:
            print("Bot: Happy gardening! Goodbye! ")
            break
            
        if not query.strip():
            continue

        print("Thinking...")
        try:
            # Generate response
            response = chain.invoke(query)
            print(f"\n Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_bot()