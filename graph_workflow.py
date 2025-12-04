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
    hallucination_status: str
    hallucination_retries: int


# --- 구조화된 출력 데이터 모델 ---

class GradeDocuments(BaseModel):
    """검색된 문서의 관련성 평가 스키마"""
    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'로 평가"
    )

class GradeHallucination(BaseModel):
    """생성된 답변의 환각 여부 평가 스키마 (Groundedness Check)"""
    binary_score: str = Field(
        description="답변이 문서에 기반하여 사실적으로 작성되었으면 'yes', 아니면 'no'"
    )


def create_rag_graph(retriever):
    """Retriever가 주입된 LangGraph 앱 생성"""

    # --- 노드 함수 정의 ---
    
    def query_decomposition_node(state: GraphState) -> GraphState:
        question = state["question"]
        prompt = ChatPromptTemplate.from_template(
            "질문을 검색하기 좋은 2개의 한국어 하위 질문으로 분리해줘. "
            "결과는 줄바꿈으로 구분해.\n질문: {question}"
        )
        chain = prompt | LLM_MODEL | StrOutputParser()
        response = chain.invoke({"question": question})
        sub_queries = [q.strip() for q in response.split("\n") if q.strip()]
        # 재시도 횟수 초기화
        return {"sub_queries": sub_queries, "hallucination_retries": 0}

    def retrieval_node(state: GraphState) -> GraphState:
        sub_queries = state.get("sub_queries", [])
        all_docs: List[str] = []

        for q in sub_queries:
            docs = retriever.invoke(q)
            for d in docs:
                all_docs.append(d.page_content)

        unique_docs = list(set(all_docs))
        return {"documents": unique_docs}

    def grade_documents_node(state: GraphState) -> GraphState:
        question = state["question"]
        documents = state.get("documents", [])
        
        structured_llm_grader = LLM_MODEL.with_structured_output(GradeDocuments)
        
        system_msg = (
            "당신은 검색된 문서가 사용자의 질문과 관련이 있는지 평가하는 채점자입니다. "
            "문서에 질문과 관련된 키워드나 의미가 포함되어 있다면 'yes'로 평가하세요. "
            "엄격할 필요는 없습니다. 관련성이 조금이라도 있다면 'yes'를 주세요."
        )
        grade_prompt = ChatPromptTemplate.from_messages(
            [("system", system_msg), ("human", "질문: {question}\n\n문서: {document}")]
        )
        retrieval_grader = grade_prompt | structured_llm_grader
        
        filtered_docs = []
        for doc in documents:
            score = retrieval_grader.invoke({"question": question, "document": doc})
            if score.binary_score.lower() == "yes":
                filtered_docs.append(doc)
        
        return {"documents": filtered_docs}

    def generate_node(state: GraphState) -> GraphState:
        question = state["question"]
        documents = state.get("documents", [])
        context = "\n\n".join(documents)

        retries = state.get("hallucination_retries", 0)
        guidance = ""
        if retries > 0:
            guidance = "이전 답변이 문서 내용을 정확히 반영하지 못했습니다. 반드시 문서에 있는 내용만 사용하여 다시 작성하세요."

        prompt = ChatPromptTemplate.from_template(
            "아래 문서를 바탕으로 질문에 대해 연구원에게 보고하듯 상세히 답변해줘.\n"
            "만약 문서에 없는 내용이라면 지어내지 말고 모른다고 답해.\n"
            "{guidance}\n\n"
            "[문서]\n{context}\n\n[질문]\n{question}"
        )
        chain = prompt | LLM_MODEL | StrOutputParser()
        generation = chain.invoke({"context": context, "question": question, "guidance": guidance})
        return {"generation": generation}

    def web_search_node(state: GraphState) -> GraphState:
        try:
            tool = TavilySearchResults(k=3)
            docs = tool.invoke({"query": state["question"]})
            web_content = [d["content"] for d in docs]
            return {"documents": web_content}
        except Exception:
            return {"documents": ["웹 검색 도구를 사용할 수 없습니다."]}

    def hallucination_check_node(state: GraphState) -> GraphState:
        """답변이 문서에 근거했는지 검사하고 재시도 횟수 업데이트"""
        documents = state.get("documents", [])
        generation = state["generation"]
        context = "\n\n".join(documents)
        
        # 현재 재시도 횟수 가져오기
        retries = state.get("hallucination_retries", 0)

        structured_llm_grader = LLM_MODEL.with_structured_output(GradeHallucination)

        system_msg = (
            "당신은 AI 답변이 참조 문서(Context)에 기반했는지 검증하는 채점자입니다. "
            "답변이 문서의 내용으로 뒷받침된다면 'yes', 문서에 없는 내용을 지어냈다면 'no'로 평가하세요."
            "만약 문서에 정보가 없어서 '모른다'고 답한 경우에도 'yes'(사실 기반)로 평가하세요."
        )
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("human", "문서: {context}\n\n답변: {generation}")
            ]
        )
        
        hallucination_grader = grade_prompt | structured_llm_grader
        score = hallucination_grader.invoke({"context": context, "generation": generation})
        
        status = score.binary_score.lower()
        
        if status == "no":
            retries += 1
            
        return {"hallucination_status": status, "hallucination_retries": retries}

    # --- 라우팅 함수들 ---

    def decide_retrieval_route(state: GraphState) -> str:
        documents = state.get("documents", [])
        if not documents:
            return "web_search"
        return "generate"

    def decide_hallucination_route(state: GraphState) -> str:
        """
        1. status == 'yes' (통과) -> END
        2. status == 'no' (환각) 이고, retries > 3 (최대 시도 초과) -> END (루프 방지)
        3. status == 'no' (환각) 이고, retries <= 3 -> generate (재시도)
        """
        status = state.get("hallucination_status", "yes")
        retries = state.get("hallucination_retries", 0)
        
        if status == "yes":
            return "end"
        elif retries > 3:
            # 최대 재시도 횟수를 초과하면 강제로 종료 (또는 web_search로 보낼 수도 있음)
            return "end" 
        else:
            return "generate"

    # --- 그래프 조립 ---
    workflow = StateGraph(GraphState)
    
    workflow.add_node("decompose", query_decomposition_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "grade")
    
    workflow.add_conditional_edges(
        "grade",
        decide_retrieval_route,
        {"web_search": "web_search", "generate": "generate"},
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "hallucination_check")
    
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_hallucination_route,
        {
            "end": END,
            "generate": "generate"
        }
    )

    return workflow.compile()