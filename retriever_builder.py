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

# PDF 파일 경로를 입력받아 최종 retriever 객체를 생성
def build_retriever(file_path: str):
    # PDF 파일을 기반으로 BM25 + Vector + RAPTOR + Reranker가 결합된 Retriever 생성

    # Streamlit 상태 표시(status) 영역 생성
    with st.status("📄 문서를 분석하고 인덱스를 생성하는 중...", expanded=True) as status:
        # 1. PDF 로딩
        st.write("1. PDF 문서 로드 중...")
        loader = PyMuPDFLoader(file_path)   # PDF를 로더로 읽어들임
        docs = loader.load()                # 페이지 별 Document 리스트 생성

        # 2. 텍스트 분할 / 청킹
        st.write("2. 텍스트 분할 및 청킹 수행 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,     # 각 청크의 최대 길이
            chunk_overlap=50,   # 문맥 유지를 위해 앞뒤로 일정 길이 겹침
        )
        splits = text_splitter.split_documents(docs)    # PDF 전체 페이지를 chunk 기반 Document 리스트로 변환

        if not splits:
            # PDF가 스캔본(이미지) 등으로 텍스트 추출이 불가한 경우 실행됨
            status.update(
                label="⚠️ 문서에서 텍스트를 찾지 못했습니다.",
                state="error",
                expanded=True,
            )
            return None

        # 3. Dense Vector Index (원문 청크 기반)
        st.write("3. Vector Index (Dense, 원문 청크) 생성 중...")
        vectorstore = FAISS.from_documents(splits, EMBEDDING_MODEL) # splits의 page_content를 임베딩 → 벡터 DB 구축
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # 검색 시 k=5개의 문서를 벡터 기반으로 가져오는 retriever 생성

        # 4. Sparse Index (BM25, 키워드 기반)
        st.write("4. BM25 Index (Sparse, 키워드 매칭) 생성 중...")
        # 텍스트 기반 키워드 검색
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5    # 검색 개수 설정

        # 5. RAPTOR 스타일 계층 요약 인덱스
        st.write("5. RAPTOR 스타일 계층 요약 인덱스 생성 중...")
        
        # build_raptor_retriever는 그룹 단위 요약 summary 문서를 만들어 summary-only 벡터 인덱스를 구축하는 retriever를 반환
        raptor_retriever = build_raptor_retriever(
            docs=splits,
            group_size=8,   # 8개 청크를 하나의 요약 노드로 묶기
            top_k=5,        # 검색 시 요약 노드 5개 반환
        )

        # 6. Ensemble Retriever 구성
         # BM25 / Vector / RAPTOR 각각 서로 다른 성질의 검색 기법이므로 가중치를 부여하여 결합 → Hybrid Retrieval
        st.write("6. Ensemble Retriever 구성 (BM25 + Vector + RAPTOR)...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever, raptor_retriever],
            weights=[0.3, 0.3, 0.4],    # RAPTOR에 더 높은 우선순위를 준 구조
        )

        # 7. Cross-Encoder Reranker로 최종 재순위화
        st.write("7. Cross-Encoder Reranker로 최종 재순위화 설정...")
        # CrossEncoderReranker는 query와 문서 pair를 입력으로 받아 더 정확한 relevance를 계산
        compressor = CrossEncoderReranker(
            model=RERANKER_MODEL,
            top_n=3,  # 최종적으로 남길 문서 수 (3개만 남김)
        )

        # 최종적으로 **가장 의미 있는 문서만 반환**하는 retriever로 변환
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,         # CrossEncoder reranker
            base_retriever=ensemble_retriever,  # Hybrid Retriever
        )

        # 상태 업데이트 (완료)
        status.update(
            label="✅ RAG Retriever 구축 완료!",
            state="complete",
            expanded=False,
        )

    # LangGraph에 전달되는 최종 retriever
    return final_retriever
