import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def initialize_models():
    """모델 로드는 시간이 걸리므로 캐싱 처리"""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    llm_model = ChatOpenAI(model="gpt-4o", temperature=0)
    # Reranker 모델 (최초 실행 시 다운로드)
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    return embedding_model, llm_model, reranker_model

EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL = initialize_models()
