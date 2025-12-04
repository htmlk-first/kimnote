from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from models import EMBEDDING_MODEL, LLM_MODEL


def _chunk_into_groups(docs: List[Document], group_size: int) -> List[List[Document]]:
    """문서 리스트를 group_size 단위로 묶어서 2차원 리스트로 반환."""
    return [docs[i : i + group_size] for i in range(0, len(docs), group_size)]


def _summarize_group(docs: List[Document]) -> str:
    """여러 청크를 하나의 요약 텍스트로 압축."""
    # group 내 모든 청크 내용을 이어붙임
    combined_text = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(
        (
            "다음은 논문 일부에서 추출한 여러 텍스트 조각입니다.\n"
            "이 조각들이 담고 있는 핵심 내용을 연구자가 빠르게 파악할 수 있도록 "
            "간결하지만 정보 손실이 최소화된 한국어 요약을 작성해 주세요.\n\n"
            "[텍스트 조각들]\n{content}"
        )
    )
    chain = prompt | LLM_MODEL | StrOutputParser()
    summary = chain.invoke({"content": combined_text})

    return summary


def build_raptor_retriever(
    docs: List[Document],
    group_size: int = 8,
    top_k: int = 5,
):
    """
    간단한 RAPTOR 스타일의 계층 요약 기반 Retriever를 생성.

    - docs: 이미 청킹된 Document 리스트 (예: RecursiveCharacterTextSplitter 결과)
    - group_size: 몇 개의 청크를 하나의 상위 요약 노드로 묶을지
    - top_k: 검색 시 반환할 요약 노드 개수
    """

    if not docs:
        raise ValueError("RAPTOR Retriever를 생성할 문서가 비어 있습니다.")

    # 1) 청크들을 group_size 단위로 묶기
    grouped_docs = _chunk_into_groups(docs, group_size=group_size)

    # 2) 각 그룹을 LLM으로 요약하여 상위 노드(요약 문서) 생성
    summary_documents: List[Document] = []
    for idx, group in enumerate(grouped_docs):
        summary_text = _summarize_group(group)

        # 첫 번째 청크의 메타데이터를 일부 상속 (source, page 등)
        base_meta = dict(group[0].metadata) if group[0].metadata else {}
        base_meta.update(
            {
                "raptor_level": "summary",
                "raptor_group_index": idx,
                # 필요하면 child 인덱스/내용을 더 추가할 수 있음
                # "child_count": len(group),
            }
        )

        summary_doc = Document(
            page_content=summary_text,
            metadata=base_meta,
        )
        summary_documents.append(summary_doc)

    # 3) 상위 요약 문서들로만 벡터스토어 생성
    #    → RAPTOR의 "상위 레벨" 인덱스 역할
    vectorstore = FAISS.from_documents(summary_documents, EMBEDDING_MODEL)

    # 4) Retriever 형태로 반환 (EnsembleRetriever에 그대로 넣을 수 있음)
    raptor_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    return raptor_retriever
