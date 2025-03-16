import streamlit as st
from core.llm import initialize_llm
from embedding.llama_embedder import LLamaEmbedder
from vector_store.pinecone_store import PineconeStore
from core.retriever import RetrievalChain
from ui.components import initialize_ui, display_chat_history
from utils.cache import initialize_cache
from utils.moderator import moderate_prompt
from config.settings import settings
from langchain_ollama import OllamaEmbeddings

def initialize_retrieval_chain():
    """Initialize the retrieval chain with LLM, embeddings, and vector store"""
    # Initialize LLM
    llm = initialize_llm()
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=settings.MODEL_NAME, base_url=settings.OLLAMA_BASE_URL)
    
    # embedder.embeddings = embeddings
    # Initialize vector store and retriever
    vector_store = PineconeStore(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME
    )
    retriever = vector_store.create_langchain_retriever(embeddings)
    
    # Create retrieval chain
    return RetrievalChain(llm, retriever)

def main():
    initialize_cache()
    initialize_ui()
    
    # Initialize the retrieval chain
    if "retrieval_chain" not in st.session_state:
        st.session_state.retrieval_chain = initialize_retrieval_chain()

    # Initialize chat history
    if "message_log" not in st.session_state:
        st.session_state.message_log = [{
            "role": "ai",
            "content": "Hi! I'm your HR Assistant. How can I help you today? 📚"
        }]

    # Display chat history
    display_chat_history(st.session_state.message_log)
    
    # Get user input
    user_query = st.chat_input("Ask me anything about HR policies...")

    if user_query:
        # Moderate the input
        is_safe, moderation_message = moderate_prompt(user_query)
        
        if not is_safe:
            st.error(moderation_message)
            return

        # Add user message to history
        st.session_state.message_log.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and display AI response
        with st.chat_message("ai"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Searching documents...", unsafe_allow_html=True)
            
            # Get response from retrieval chain
            response = st.session_state.retrieval_chain.get_response(user_query)
            
            # Update display and message history
            response_placeholder.markdown(response)
            st.session_state.message_log.append({"role": "ai", "content": response})

if __name__ == "__main__":
    main()