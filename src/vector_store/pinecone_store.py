from pinecone import Pinecone, Index
from typing import List, Dict
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document

class PineconeStore:
    def __init__(self, api_key: str, index_name: str = "test"):
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Get or create index
        try:
            self.index: Index = self.pc.Index(index_name)
        except Exception as e:
            # Create index if it doesn't exist
            self.pc.create_index(
                name=index_name,
                dimension=3072,  # adjust based on your LLama embeddings dimension
                metric="cosine"
            )
            self.index: Index = self.pc.Index(index_name)

    def create_langchain_retriever(self, embeddings: Embeddings, k: int = 3):
        # Initialize Pinecone for Langchain
        vectorstore = PineconeVectorStore(
            self.index,
            embeddings,  # Pass the embedding function
            "content"  # This specifies which metadata field contains the document text
        )
        
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": 0.7,
                # "filter": {"type": "metadata"}  # Add any specific filters you need
            }
        )