import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from models import EMBEDDING_MODEL, RERANKER_MODEL
from raptor_builder import build_raptor_retriever


def build_retriever(file_path: str):
    """PDF 파일을 기반으로 BM25 + Vector + RAPTOR + Reranker가 결합된 Retriever 생성"""

    with st.status("📄 문서를 분석하고 인덱스를 생성하는 중...", expanded=True) as status:
        # 1. PDF 로딩
        st.write("1. PDF 문서 로드 중...")
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # 2. 텍스트 분할 / 청킹
        st.write("2. 텍스트 분할 및 청킹 수행 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        splits = text_splitter.split_documents(docs)

        if not splits:
            status.update(
                label="⚠️ 문서에서 텍스트를 찾지 못했습니다.",
                state="error",
                expanded=True,
            )
            return None

        # 3. Dense Vector Index (원문 청크 기반)
        st.write("3. Vector Index (Dense, 원문 청크) 생성 중...")
        vectorstore = FAISS.from_documents(splits, EMBEDDING_MODEL)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 4. Sparse Index (BM25, 키워드 기반)
        st.write("4. BM25 Index (Sparse, 키워드 매칭) 생성 중...")
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5

        # 5. RAPTOR 스타일 계층 요약 인덱스
        st.write("5. RAPTOR 스타일 계층 요약 인덱스 생성 중...")
        # group_size: 몇 개의 청크를 하나의 상위 요약 노드로 묶을지
        # top_k: 질문당 반환할 상위 요약 노드 개수
        raptor_retriever = build_raptor_retriever(
            docs=splits,
            group_size=8,
            top_k=5,
        )

        # 6. Ensemble Retriever 구성
        st.write("6. Ensemble Retriever 구성 (BM25 + Vector + RAPTOR)...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever, raptor_retriever],
            # 예시 비율: RAPTOR에 약간 더 가중치
            weights=[0.3, 0.3, 0.4],
        )

        # 7. Cross-Encoder Reranker로 최종 재순위화
        st.write("7. Cross-Encoder Reranker로 최종 재순위화 설정...")
        compressor = CrossEncoderReranker(
            model=RERANKER_MODEL,
            top_n=3,  # 최종적으로 남길 상위 문서 수
        )

        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )

        status.update(
            label="✅ RAG Retriever 구축 완료!",
            state="complete",
            expanded=False,
        )

    return final_retriever
