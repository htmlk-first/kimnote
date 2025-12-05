import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# .env 파일 로드
load_dotenv()

# ---------------------------------------------------------------------
# Streamlit의 리소스 캐싱 데코레이터
# - 모델(embedding, llm, reranker)은 로드 비용이 크므로 최초 한 번만 로드
# - 이후 실행에서는 같은 객체를 재사용하여 속도 향상 + 비용 절감
# ---------------------------------------------------------------------
@st.cache_resource
def initialize_models():
    """Embedding / LLM / Reranker 모델 로드 (캐싱)"""
    
    # 1) Embedding Model
    # OpenAI의 최신 임베딩 모델(저비용 + 고품질)
    # - 목적: 문서를 벡터 공간에 매핑하기 위함 (벡터 검색용)
    # - retriever_builder.py의 FAISS 인덱스 생성 단계에서 사용됨
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2) LLM Model
    # "gpt-4o" OpenAI 모델을 사용하며 여러 노드에서 공용으로 사용됨
    # - Query decomposition (질문 분해)
    # - 문서 요약 (RAPTOR summary 생성)
    # - 문서 필터링 grader
    # - hallucination check grader
    # - 최종 답변 생성
    # temperature=0 → 일관성 있는 답변 생성 (연구/문서 기반 환경에 적합)
    llm_model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 3) Cross-Encoder Reranker (최초 실행 시 다운로드)
    # BAAI의 bge-reranker 모델
    # - query와 document를 한 번에 인코딩하여 relevance score를 정밀 계산
    # - 벡터 검색(FAISS)나 BM25보다 정밀도(precision)가 훨씬 높음
    #
    # reranker_builder.py에서 ContextualCompressionRetriever에 사용됨
    # → 최종적으로 "질문에 정말 맞는 상위 3개 문서"만 남겨주는 역할
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    return embedding_model, llm_model, reranker_model

EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL = initialize_models()
