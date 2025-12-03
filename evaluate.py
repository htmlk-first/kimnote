import os
from dotenv import load_dotenv
from langchain.smith import RunEvalConfig
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI

# 1. 기존 RAG 로직 임포트 (app.py 또는 main.py에서 필요한 함수 가져오기)
# 주의: app.py에서 streamlit 관련 코드가 실행되지 않도록 구조를 분리하거나,
# 필요한 함수(build_retriever, create_rag_graph)만 복사해오세요.
# 여기서는 편의상 app.py의 로직을 함수로 호출한다고 가정합니다.
from main import build_retriever, workflow # main.py에서 workflow(app) 객체를 가져옴

load_dotenv()

# 2. 평가 대상 타겟 정의 (RAG 체인/그래프)
# 실제 평가 시에는 PDF를 로드한 상태의 그래프를 사용해야 합니다.
# 테스트를 위해 로컬의 특정 PDF를 로드하여 retriever를 빌드합니다.
pdf_path = "data/RAG Project 계획서.pdf" # 평가에 사용할 실제 PDF 파일 경로
retriever = build_retriever(pdf_path)

# LangGraph 앱 컴파일 (검색기가 주입된 상태여야 함)
# 만약 main.py가 전역변수 RETRIEVER를 쓴다면 미리 설정
import main
main.RETRIEVER = retriever
target_app = main.app # 컴파일된 LangGraph 앱

# 3. 래퍼 함수: 데이터셋의 입력을 받아 그래프를 실행하고 결과 반환
def target_wrapper(inputs):
    response = target_app.invoke({"question": inputs["question"]})
    return {"output": response["generation"]}

# 4. 평가 설정 (채점 기준)
eval_config = RunEvalConfig(
    evaluators=[
        # QA Correctness: 질문, 정답, 생성된 답변을 비교하여 "정확함/부정확함" 판단
        "cot_qa", 
    ],
    # 평가에 사용할 LLM (심판)
    eval_llm=ChatOpenAI(model="gpt-4o", temperature=0)
)

# 5. 평가 실행
print("Running Evaluation on LangSmith...")
chain_results = evaluate(
    target_wrapper,
    data="UAV Research QA Test", # 2단계에서 만든 데이터셋 이름
    evaluators=[eval_config],
    experiment_prefix="UAV-RAG-Experiment",
    metadata={"version": "1.0", "description": "UAV 논문 RAG 성능 평가"},
)

print("\nEvaluation Complete! Check LangSmith Dashboard.")