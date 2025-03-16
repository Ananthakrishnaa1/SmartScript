import streamlit as st
import re
from core.llm import initialize_llm
from core.prompt_templates import get_system_prompt, build_prompt_chain
from ui.components import initialize_ui, display_chat_history
from utils.cache import initialize_cache
from utils.moderator import moderate_prompt
from langchain_core.output_parsers import StrOutputParser

def main():
    initialize_cache()
    initialize_ui()
    
    llm_engine = initialize_llm()
    system_prompt = get_system_prompt()

    if "message_log" not in st.session_state:
        st.session_state.message_log = [{
            "role": "ai",
            "content": "Hi! I'm your Personal ChatBot. How can I help you code today? 💻"
        }]

    display_chat_history(st.session_state.message_log)
    user_query = st.chat_input("Type your coding question here...")

    if user_query:
        is_safe, moderation_message = moderate_prompt(user_query)
        
        if not is_safe:
            st.error(moderation_message)
            return

        st.session_state.message_log.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("ai"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...", unsafe_allow_html=True)
            
            prompt_chain = build_prompt_chain(st.session_state.message_log, system_prompt)
            processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
            
            full_response = ""
            for chunk in processing_pipeline.stream({}):
                full_response += chunk
            
            cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
            response_placeholder.markdown(cleaned_response)
            st.session_state.message_log.append({"role": "ai", "content": cleaned_response})

if __name__ == "__main__":
    main()