import os
import tempfile
from typing import List, TypedDict

import streamlit as st
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END, START

# 1. 환경 설정 로드
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title="UAV 연구 보조 RAG", page_icon="🚁")
st.title("UAV 연구 보조 Agentic RAG")

# ==============================================================================
# [Part 1] 캐싱된 RAG 시스템 빌더
# ==============================================================================


@st.cache_resource
def initialize_models():
    """모델 로드는 시간이 걸리므로 캐싱 처리"""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    llm_model = ChatOpenAI(model="gpt-4o", temperature=0)
    # Reranker 모델 (최초 실행 시 다운로드)
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    return embedding_model, llm_model, reranker_model


EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL = initialize_models()


def build_retriever(file_path: str):
    """PDF 파일을 기반으로 Retriever 생성"""
    with st.status("📄 문서를 분석하고 인덱스를 생성하는 중...", expanded=True) as status:
        st.write("1. PDF 문서 로드 중...")
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        st.write("2. 텍스트 분할 및 청킹...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        texts = [d.page_content for d in splits]

        if not texts:
            status.update(
                label="⚠️ 문서에서 텍스트를 찾지 못했습니다.", state="error", expanded=True
            )
            return None

        st.write("3. Vector Index (Dense) 생성 중...")
        vectorstore = FAISS.from_texts(texts, EMBEDDING_MODEL)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        st.write("4. BM25 Index (Sparse) 생성 중...")
        bm25_retriever = BM25Retriever.from_texts(texts)
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


# ==============================================================================
# [Part 2] LangGraph Workflow 정의
# ==============================================================================


class GraphState(TypedDict, total=False):
    question: str
    documents: List[str]
    generation: str
    sub_queries: List[str]


def create_rag_graph(retriever):
    """Retriever가 주입된 LangGraph 앱 생성"""

    # --- 노드 함수 ---
    def query_decomposition_node(state: GraphState) -> GraphState:
        question = state["question"]
        prompt = ChatPromptTemplate.from_template(
            "질문을 검색하기 좋은 2개의 한국어 하위 질문으로 분리해줘. "
            "결과는 줄바꿈으로 구분해.\n질문: {question}"
        )
        chain = prompt | LLM_MODEL | StrOutputParser()
        response = chain.invoke({"question": question})
        sub_queries = [q.strip() for q in response.split("\n") if q.strip()]
        return {"sub_queries": sub_queries}

    def retrieval_node(state: GraphState) -> GraphState:
        sub_queries = state.get("sub_queries", [])
        all_docs: List[str] = []

        for q in sub_queries:
            docs = retriever.invoke(q)
            for d in docs:
                all_docs.append(d.page_content)

        # 중복 제거
        unique_docs = list(set(all_docs))
        return {"documents": unique_docs}

    def grade_documents_node(state: GraphState) -> GraphState:
        # 간소화된 평가 로직 (그대로 통과)
        return state

    def generate_node(state: GraphState) -> GraphState:
        question = state["question"]
        documents = state.get("documents", [])
        context = "\n\n".join(documents)

        prompt = ChatPromptTemplate.from_template(
            "아래 문서를 바탕으로 질문에 대해 연구원에게 보고하듯 상세히 답변해줘.\n\n"
            "[문서]\n{context}\n\n[질문]\n{question}"
        )
        chain = prompt | LLM_MODEL | StrOutputParser()
        generation = chain.invoke({"context": context, "question": question})
        return {"generation": generation}

    def web_search_node(state: GraphState) -> GraphState:
        try:
            tool = TavilySearchResults(k=3)
            docs = tool.invoke({"query": state["question"]})
            web_content = [d["content"] for d in docs]
            return {"documents": web_content}
        except Exception:
            return {
                "documents": ["웹 검색 도구를 사용할 수 없습니다 (API Key 확인 필요)."]
            }

    def decide_route(state: GraphState) -> str:
        documents = state.get("documents", [])
        if not documents:
            return "web_search"
        return "generate"

    # --- 그래프 조립 ---
    workflow = StateGraph(GraphState)
    workflow.add_node("decompose", query_decomposition_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("web_search", web_search_node)

    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        decide_route,
        {"web_search": "web_search", "generate": "generate"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ==============================================================================
# [Part 3] Streamlit UI 구성
# ==============================================================================

# 세션 상태 초기화
if "messages" not in st.session_state:
    # 간단한 dict 구조로 관리: {"role": "user"|"assistant", "content": str}
    st.session_state["messages"] = []

if "rag_app" not in st.session_state:
    st.session_state["rag_app"] = None


# 사이드바: 파일 업로드 및 설정
with st.sidebar:
    st.header("📂 문서 업로드")
    uploaded_file = st.file_uploader("연구 논문(PDF)을 업로드하세요", type=["pdf"])

    if uploaded_file:
        # 임시 파일로 저장 (PyMuPDFLoader는 경로가 필요함)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 파일이 변경되었을 때만 재빌드
        if (
            "current_file" not in st.session_state
            or st.session_state["current_file"] != uploaded_file.name
        ):
            retriever = build_retriever(tmp_file_path)
            if retriever:
                st.session_state["rag_app"] = create_rag_graph(retriever)
                st.session_state["current_file"] = uploaded_file.name
                st.success("RAG 시스템 준비 완료!")
            else:
                st.error("RAG 시스템 생성에 실패했습니다. PDF 내용을 확인해주세요.")

        # 임시 파일 삭제
        os.remove(tmp_file_path)

    st.divider()
    if st.button("대화 내용 초기화"):
        st.session_state["messages"] = []
        st.rerun()


# 채팅 히스토리 출력 함수
def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])


def add_history(role: str, content: str):
    st.session_state["messages"].append({"role": role, "content": content})


# 메인 화면 렌더링
print_history()

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지 추가 및 출력
    add_history("user", user_input)
    st.chat_message("user").write(user_input)

    # 2. AI 응답 생성
    if st.session_state["rag_app"] is None:
        st.warning("먼저 왼쪽 사이드바에서 PDF 파일을 업로드해주세요.")
    else:
        with st.chat_message("assistant"):
            chat_container = st.empty()

            # LangGraph 실행 및 결과 스트리밍 시뮬레이션
            inputs = {"question": user_input}
            app = st.session_state["rag_app"]

            # 단계별 진행 상황 표시
            with st.status("AI가 생각 중...", expanded=True) as status:
                final_answer = ""

                # stream()을 사용하여 노드 진행 상황을 볼 수 있음
                for output in app.stream(inputs):
                    for key, value in output.items():
                        st.write(f"🚩 **{key}** 단계 완료")
                        if key == "generate":
                            final_answer = value["generation"]

                status.update(label="답변 생성 완료", state="complete", expanded=False)

            # 최종 답변 출력
            chat_container.markdown(final_answer)
            add_history("assistant", final_answer)
