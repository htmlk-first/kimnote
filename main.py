import os
import tempfile
from typing import List, TypedDict
import streamlit as st
from dotenv import load_dotenv

from retriever_builder import build_retriever
from graph_workflow import create_rag_graph

# 1. 환경 설정 로드
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title="UAV 연구 보조 RAG", page_icon="🚁")
st.title("UAV 연구 보조 Agentic RAG")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "rag_app" not in st.session_state:
    st.session_state["rag_app"] = None

if "current_file_hash" not in st.session_state:
    st.session_state["current_file_hash"] = None

# 채팅 히스토리 출력 함수
def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

def add_history(role: str, content: str):
    st.session_state["messages"].append({"role": role, "content": content})


# 사이드바: 파일 업로드 및 설정
with st.sidebar:
    st.header("📂 문서 업로드")
    uploaded_file = st.file_uploader("연구 논문(PDF)을 업로드하세요", type=["pdf"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_hash = hash(file_bytes)

        # 내용이 바뀐 경우에만 retriever / graph 재생성
        if st.session_state["current_file_hash"] != file_hash:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name

            retriever = build_retriever(tmp_file_path)
            os.remove(tmp_file_path)

            if retriever:
                st.session_state["rag_app"] = create_rag_graph(retriever)
                st.session_state["current_file_hash"] = file_hash
                st.success("RAG 시스템 준비 완료!")
            else:
                st.session_state["rag_app"] = None
                st.error("RAG 시스템 생성에 실패했습니다. PDF 내용을 확인해주세요.")

    st.divider()
    if st.button("대화 내용 초기화"):
        st.session_state["messages"] = []
        st.rerun()


# 메인 화면 렌더링
print_history()

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요..."):
    add_history("user", user_input)
    st.chat_message("user").write(user_input)

    if st.session_state["rag_app"] is None:
        st.warning("먼저 왼쪽 사이드바에서 PDF 파일을 업로드해주세요.")
    else:
        with st.chat_message("assistant"):
            chat_container = st.empty()

            inputs = {"question": user_input}
            app = st.session_state["rag_app"]

            with st.status("AI가 생각 중...", expanded=True) as status:
                final_answer = ""

                for output in app.stream(inputs):
                    for key, value in output.items():
                        st.write(f"🚩 **{key}** 단계 완료")
                        if key == "generate":
                            final_answer = value["generation"]

                status.update(label="답변 생성 완료", state="complete", expanded=False)

            chat_container.markdown(final_answer)
            add_history("assistant", final_answer)
