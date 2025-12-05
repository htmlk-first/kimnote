import os
import tempfile
from typing import List, TypedDict
import streamlit as st
from dotenv import load_dotenv

from retriever_builder import build_retriever
from graph_workflow import create_rag_graph

# 1. 환경 설정 로드 (.env 파일)
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(
    page_title="UAV 연구 보조 RAG", 
    page_icon="🚁")
st.title("UAV 연구 보조 Agentic RAG")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []   # 채팅 히스토리를 저장할 리스트 초기화

if "rag_app" not in st.session_state:
    st.session_state["rag_app"] = None  # LangGraph로 컴파일된 RAG 앱 객체를 저장할 슬롯

if "current_file_hash" not in st.session_state:
    st.session_state["current_file_hash"] = None    # 현재 로드된 PDF 파일 내용의 해시값을 저장

# 채팅 히스토리 출력 함수
def print_history():
    # 세션에 저장된 모든 메시지를 순서대로 화면에 출력
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

def add_history(role: str, content: str):
    # 새로운 채팅 메시지를 세션 상태의 히스토리 리스트에 추가
    st.session_state["messages"].append({"role": role, "content": content})


# 사이드바: 파일 업로드 및 설정
with st.sidebar:    # 화면 왼쪽에 위치한 사이드바 영역 정의
    st.header("📂 문서 업로드") # 사이드바 상단에 섹션 헤더 출력
    uploaded_file = st.file_uploader(
        "연구 논문(PDF)을 업로드하세요", 
        type=["pdf"])   # 허용 파일 확장자: pdf만

    if uploaded_file:   # 사용자가 PDF 파일을 하나 업로드했을 때만 실행
        file_bytes = uploaded_file.getvalue()   # 업로드된 파일의 바이너리 내용을 메모리로 읽기
        file_hash = hash(file_bytes)            # 파일 내용의 해시값 계산   

        # 내용이 바뀐 경우에만 retriever / graph 재생성
        if st.session_state["current_file_hash"] != file_hash:
            # 임시 파일 생성 (PyMuPDFLoader 등은 파일 경로를 필요로 하기 때문에)
            with tempfile.NamedTemporaryFile(
                delete=False,   # Streamlit 프로세스에서 명시적으로 삭제하기 전까지 유지
                suffix=".pdf"   # 파일 확장자를 .pdf로 지정
            ) as tmp_file:
                tmp_file.write(file_bytes)      # 업로드된 바이너리 내용을 임시 파일에 기록
                tmp_file_path = tmp_file.name   # 생성된 임시 파일의 실제 경로

            retriever = build_retriever(tmp_file_path)  # PDF 파일 경로를 넘겨 RAG용 Retriever 생성
            os.remove(tmp_file_path)                    # 더 이상 필요 없는 임시 파일 삭제

            if retriever:
                st.session_state["rag_app"] = create_rag_graph(retriever)   # Retriever가 정상 생성되었다면, 이를 기반으로 LangGraph RAG 앱 생성
                st.session_state["current_file_hash"] = file_hash           # 현재 세션에 이 파일의 해시값 저장 (다음 업로드 때 내용 변경 여부 비교용)
                st.success("RAG 시스템 준비 완료!")                           # 사용자에게 준비 완료 메시지 출력
            else:
                # Retriever 생성에 실패했을 경우 RAG 앱 초기화 및 에러 알림
                st.session_state["rag_app"] = None
                st.error("RAG 시스템 생성에 실패했습니다. PDF 내용을 확인해주세요.")

    st.divider()    # 사이드바에 시각적 구분선 추가
    
    # "대화 내용 초기화" 버튼을 사이드바에 생성
    if st.button("대화 내용 초기화"):
        st.session_state["messages"] = []   # 세션에 저장된 채팅 히스토리를 전부 삭제
        st.rerun()                          # 앱을 다시 실행시켜 화면을 초기 상태로 리렌더링


# 메인 화면 렌더링
print_history() # 지금까지의 채팅 히스토리를 메인 채팅 영역에 재출력

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요..."):
    # 사용자가 채팅 입력창에 질문을 입력하고 엔터를 치면 이 블록이 실행됨
    add_history("user", user_input)             # 세션 히스토리에 사용자 메시지 추가
    st.chat_message("user").write(user_input)   # 화면에 사용자의 질문 출력

    if st.session_state["rag_app"] is None:
        # 아직 PDF가 업로드되지 않아 RAG 앱이 준비되지 않은 상태라면 경고 메시지 출력
        st.warning("먼저 왼쪽 사이드바에서 PDF 파일을 업로드해주세요.")
    else:
        # RAG 앱이 준비된 상태에서만 AI 응답 생성 진행
        with st.chat_message("assistant"):
            # assistant 역할의 메시지 컨테이너 생성
            chat_container = st.empty()  # 나중에 최종 답변을 표시하기 위한 placeholder

            # LangGraph에 전달할 입력 상태: 질문 문자열만 포함
            inputs = {"question": user_input}
            app = st.session_state["rag_app"]   # 세션에 저장된 LangGraph 앱 인스턴스 가져오기

            # LangGraph의 각 노드 실행 상황을 시각적으로 보여주는 상태 표시 위젯
            with st.status("AI가 생각 중...", expanded=True) as status:
                final_answer = ""   # 최종 생성된 답변 텍스트를 담을 변수

                # LangGraph 앱의 스트리밍 실행 결과를 순차적으로 처리
                # 각 노드가 완료될 때마다 상태 표시 위젯에 업데이트
                for output in app.stream(inputs):
                    for key, value in output.items():
                        # 각 노드가 끝날 때마다 해당 노드 이름을 화면에 로그로 보여줌
                        st.write(f"🚩 **{key}** 단계 완료")
                        if key == "generate":
                            # generate 노드가 실행된 시점의 state에서 최종 답변 텍스트를 추출
                            final_answer = value["generation"]

                # 모든 노드 실행이 끝난 후 상태 위젯 라벨/상태를 업데이트
                status.update(label="답변 생성 완료", state="complete", expanded=False)

            # 위에서 준비한 placeholder(chat_container)에 최종 답변을 마크다운 형태로 출력
            chat_container.markdown(final_answer)
            # assistant 응답도 세션 히스토리에 저장하여 이후 화면 재렌더링 시 복원
            add_history("assistant", final_answer)
