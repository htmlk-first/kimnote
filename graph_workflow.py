from typing import List, TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

from models import LLM_MODEL


class GraphState(TypedDict, total=False):
    question: str
    documents: List[str]
    generation: str
    sub_queries: List[str]

# --- 1. 평가를 위한 데이터 구조 정의 (Pydantic) ---
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성 평가 스키마"""
    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'로 평가"
    )

def create_rag_graph(retriever):
    """Retriever가 주입된 LangGraph 앱 생성"""

    # --- 노드 함수 ---
    def query_decomposition_node(state: GraphState) -> GraphState:
        question = state["question"]
        # 질문 분해 프롬프트
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

        # 하위 질문들에 대해 검색 수행
        for q in sub_queries:
            docs = retriever.invoke(q)
            for d in docs:
                all_docs.append(d.page_content)

        # 중복 제거
        unique_docs = list(set(all_docs))
        return {"documents": unique_docs}

    def grade_documents_node(state: GraphState) -> GraphState:
        question = state["question"]
        documents = state.get("documents", [])
        
        # 1. LLM에 구조화된 출력(Structured Output) 바인딩
        structured_llm_grader = LLM_MODEL.with_structured_output(GradeDocuments)

        # 2. 평가 프롬프트 설정 (시스템 프롬프트로 역할 부여)
        system_msg = (
            "당신은 검색된 문서가 사용자의 질문과 관련이 있는지 평가하는 채점자입니다. "
            "문서에 질문과 관련된 키워드나 의미가 포함되어 있다면 'yes'로 평가하세요. "
            "엄격할 필요는 없습니다. 관련성이 조금이라도 있다면 'yes'를 주세요."
        )
        
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("human", "질문: {question}\n\n문서: {document}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader
        
        # 3. 문서 필터링 Loop
        filtered_docs = []
        for doc in documents:
            # 각 문서에 대해 평가 수행
            score = retrieval_grader.invoke({"question": question, "document": doc})
            
            if score.binary_score.lower() == "yes":
                filtered_docs.append(doc)
            # 'no'인 경우 리스트에 추가하지 않음 (제거됨)
        
        return {"documents": filtered_docs}

    def generate_node(state: GraphState) -> GraphState:
        question = state["question"]
        documents = state.get("documents", [])
        context = "\n\n".join(documents)

        # 답변 생성 프롬프트
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
        # 평가(Grade) 결과에 따라 다음 경로 결정
        if not documents:
            return "web_search"
        return "generate"

    # --- 그래프 조립 ---
    workflow = StateGraph(GraphState)
    # 노드 추가
    workflow.add_node("decompose", query_decomposition_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("web_search", web_search_node)
    # 엣지 연결
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "grade")
    # 조건부 엣지: grade 결과에 따라 분기
    workflow.add_conditional_edges(
        "grade",
        decide_route,
        {
            "web_search": "web_search",
            "generate": "generate"
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
