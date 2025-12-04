from typing import Optional

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from models import EMBEDDING_MODEL, RERANKER_MODEL


def build_retriever(file_path: str):
    """PDF 파일을 기반으로 Retriever 생성"""
    with st.status("📄 문서를 분석하고 인덱스를 생성하는 중...", expanded=True) as status:
        st.write("1. PDF 문서 로드 중...")
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        st.write("2. 텍스트 분할 및 청킹...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        if not splits:
            status.update(
                label="⚠️ 문서에서 텍스트를 찾지 못했습니다.",
                state="error",
                expanded=True,
            )
            return None

        st.write("3. Vector Index (Dense) 생성 중...")
        # Document 객체 그대로 사용 (metadata 보존)
        vectorstore = FAISS.from_documents(splits, EMBEDDING_MODEL)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        st.write("4. BM25 Index (Sparse) 생성 중...")
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5

        st.write("5. Ensemble 및 Reranker 설정...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6],
        )

        compressor = CrossEncoderReranker(model=RERANKER_MODEL, top_n=3)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )

        status.update(
            label="✅ RAG 시스템 구축 완료!", state="complete", expanded=False
        )

    return final_retriever
