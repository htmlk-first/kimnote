# UAV 연구 보조 Agentic RAG 시스템

이 프로젝트는 **UAV(무인항공기) 최적화 및 통신 분야의 연구 논문**을 효과적으로 분석하고 질의응답을 수행하기 위해 개발된 **Agentic RAG(검색 증강 생성)** 애플리케이션입니다.

**LangGraph**를 활용한 에이전트 워크플로우를 통해 복잡한 질문을 분해하고, **Hybrid Search(BM25 + Vector)** 및 **Reranker**를 통해 검색 정확도를 극대화했습니다.


## ✨ 주요 기능

* **PDF 논문 분석**: 사용자가 업로드한 PDF 논문을 분석하여 지식 베이스를 구축합니다.
* **고급 검색 시스템 (Advanced Retrieval)**:
    * **Ensemble Retriever**: 키워드 검색(BM25)과 의미 기반 검색(FAISS Vector)을 결합하여 UAV 분야의 전문 용어(예: SINR, Trajectory Optimization)와 문맥적 의미를 모두 포착합니다.
    * **Cross-Encoder Reranker**: 검색된 문서의 관련성을 정밀하게 재순위화하여 정확도를 높입니다.

* **에이전트 워크플로우 (LangGraph)**:
    * **Query Decomposition**: 복잡한 연구 질문을 검색 가능한 하위 질문들(Sub-queries)로 분해합니다.
    * **Adaptive Routing**: 문서 내 정보가 부족할 경우 자동으로 **웹 검색(Tavily)**으로 전환하여 최신 정보를 보완합니다.

* **직관적인 UI**: Streamlit을 활용하여 논문 업로드, 진행 상황 시각화, 채팅 인터페이스를 제공합니다.

## 🛠️ 기술 스택

* **Framework**: [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
* **UI**: [Streamlit](https://streamlit.io/)
* **LLM & Embedding**: OpenAI (GPT-4o, text-embedding-3-small)
* **Vector Store**: FAISS
* **Retriever**: RankBM25, Cross-Encoder (BAAI/bge-reranker-v2-m3)
* **Tools**: Tavily Search API (Web Search)

## 📂 프로젝트 구조

```bash
.
├── app.py               # 메인 Streamlit 애플리케이션 코드
├── requirements.txt     # 프로젝트 의존성 라이브러리 목록
├── .env                 # API Key 설정 파일 (사용자 생성 필요)
└── README.md            # 프로젝트 설명서
```

## 🚀 설치 및 실행 방법

### 1. 환경 설정

먼저 프로젝트를 실행하기 위해 Python 환경을 준비하고 필수 라이브러리를 설치합니다.

```bash
# 1. (선택사항) 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Mac/Linux
source venv/Scripts/activate  # Windows

# 2. 필수 라이브러리 설치
pip install -r requirements.txt
```

### 2. API 키 설정

프로젝트 루트 경로에 .env 파일을 생성하고, 본인의 API Key를 입력합니다.

.env 파일 내용:

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. 애플리케이션 실행

터미널에서 다음 명령어를 실행하여 Streamlit 앱을 시작합니다.

```bash
streamlit run main.py
```

### 4. 사용 방법

1. 웹 브라우저가 열리면 왼쪽 사이드바의 "문서 업로드" 버튼을 통해 분석할 PDF 논문을 업로드합니다.

2. 시스템이 문서를 분석하고 인덱싱(Vector + BM25)을 완료할 때까지 잠시 기다립니다. (상태 메시지가 표시됩니다.)

3. 하단 채팅창에 논문과 관련된 질문을 입력합니다.

4. 예시: "이 논문에서 제안하는 UAV 경로 최적화 알고리즘의 핵심은 무엇인가요?"

5. AI가 질문을 분석(Decomposition)하고 문서를 검색(Retrieval)하여 답변을 생성하는 과정을 실시간으로 확인합니다.


## 🧩 LangGraph 워크플로우 상세

이 시스템은 다음과 같은 순환형 그래프 구조로 동작합니다.

1. Decompose: 사용자의 질문을 검색하기 쉬운 형태의 하위 질문들로 분해합니다.

2. Retrieve: 분해된 질문을 바탕으로 논문 내용을 검색합니다 (Hybrid Search).

3. Grade: 검색된 문서가 질문에 답변하기 충분한지 평가합니다 (현재 버전에서는 간소화됨).

4. Decide Route:

   * 문서가 충분하면 -> Generate (답변 생성)

   * 문서가 없거나 부족하면 -> Web Search (외부 검색)

5. Generate: 수집된 정보를 종합하여 최종 답변을 생성합니다.
