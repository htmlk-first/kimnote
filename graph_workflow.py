from typing import List, TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from models import LLM_MODEL


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
