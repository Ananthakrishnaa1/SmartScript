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

# --- Authentication ---
def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] == settings.USERNAME
            and st.session_state["password"] == settings.PASSWORD
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        # Show inputs for username + password.
        st.text_input(label="Username", key="username")
        st.text_input(
            label="Password",
            type="password",
            key="password",
            on_change=password_entered,
        )
        if st.session_state["password_correct"] is False:
            st.error("😕 User not known or password incorrect")
        # Halt execution if the password not correct.
        st.stop()

    return True

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
    if not check_password():
        return
    
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
    
    # Initialize communication log
    if "communication_log" not in st.session_state:
        st.session_state.communication_log = []

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
            retrieval_chain = st.session_state.retrieval_chain
            response = retrieval_chain.get_response(user_query)
            
            # Update display and message history
            response_placeholder.markdown(response)
            st.session_state.message_log.append({"role": "ai", "content": response})
            
            # Capture communication details
            st.session_state.communication_log.append({
                "user_query": user_query,
                "response": response,
                "memory": retrieval_chain.memory.load_memory_variables({}) # Capture memory
            })

if __name__ == "__main__":
    main()