from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

def initialize_cache():
    set_llm_cache(InMemoryCache())