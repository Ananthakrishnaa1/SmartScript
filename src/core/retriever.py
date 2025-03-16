from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Any
from prompts.qa_prompts import rephrase_prompt, response_prompt

class RetrievalChain:
    def __init__(self, llm: Any, retriever: Any):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            verbose=True,
            rephrase_question=True,
            condense_question_prompt=rephrase_prompt,
            combine_docs_chain_kwargs={
                "prompt": response_prompt
            },
            return_source_documents=False
        )

    def get_response(self, query: str) -> str:
        try:
            response = self.chain.invoke({
                "question": query,
                "chat_history": self.memory.chat_memory.messages
            })
            return response["answer"]
        except Exception as e:
            return f"Error getting response: {str(e)}"