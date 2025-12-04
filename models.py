import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

@st.cache_resource
def initialize_models():
    """Embedding / LLM / Reranker 모델 로드 (캐싱)"""
    # 1) Embedding
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2) LLM
    llm_model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 3) Reranker (Cross-Encoder) (최초 실행 시 다운로드)
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    return embedding_model, llm_model, reranker_model

EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL = initialize_models()
