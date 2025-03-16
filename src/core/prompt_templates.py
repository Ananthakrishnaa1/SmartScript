from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

def get_system_prompt():
    return SystemMessagePromptTemplate.from_template(
        "You are an expert AI coding assistant integrated by Ananthakrishna. "
        "Provide concise, correct solutions and with strategic print statements "
        "for debugging. Always respond in English."
    )

def build_prompt_chain(message_log, system_prompt):
    prompt_sequence = [system_prompt]
    for msg in message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)