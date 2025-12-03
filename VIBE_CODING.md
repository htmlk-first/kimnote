# ✨ Vibe Coding: AI-Assisted Development Journey

> **"Coding is not about syntax anymore, it's about the Vibe."**

이 프로젝트는 **Vibe Coding(바이브 코딩)** 방법론을 적극적으로 도입하여 개발되었습니다. 개발자는 기획과 논리적 흐름(Vibe)을 설계하고, AI(LLM)는 이를 실제 동작하는 코드로 구현하는 협업 프로세스를 통해 완성되었습니다.

## 🌊 What is Vibe Coding?

**Vibe Coding**은 전통적인 코딩 방식인 "한 줄 한 줄 직접 작성하기"에서 벗어나, AI에게 자연어로 의도와 맥락을 전달하고 생성된 코드를 조율하는 새로운 개발 패러다임입니다.

* **Focus on Intent**: `for` 루프나 문법 오류에 집중하기보다, "이 논문을 분석해서 요약해줘"와 같은 **핵심 기능과 목적**에 집중합니다.
* **Speed & Flow**: 아이디어를 코드로 변환하는 시간이 획기적으로 단축되어, 개발의 흐름(Flow)이 끊기지 않습니다.
* **Iterative Creation**: AI가 작성한 초안을 바탕으로, 개발자가 피드백을 주며 점진적으로 완성도를 높여갑니다.

## 🚀 How We Built This (Workflow)

이 **UAV 연구 보조 RAG 시스템**은 다음과 같은 Vibe Coding 프로세스를 거쳐 탄생했습니다.

### 1. The Vibe (기획 및 의도 전달)

* **Input**: `RAG Project 계획서.pdf`라는 명확한 청사진을 AI에게 제공했습니다.
* **Prompt**: "이 계획서의 구조(RAPTOR, Ensemble, LangGraph)를 완벽하게 구현하는 코드를 짜줘."라는 강력한 의도를 전달했습니다.

### 2. The Flow (코드 생성 및 구현)

* **Data Processing**: 복잡한 논문 전처리 로직(PyMuPDF, TextSplitter)을 AI가 즉시 구현했습니다.
* **Logic Design**: LangGraph의 복잡한 노드(Node)와 엣지(Edge) 연결 구조를 AI가 설계도에 맞춰 코딩했습니다.
* **Hybrid Search**: BM25와 Vector Search를 결합하는 고난도 검색 로직을 단 몇 번의 대화로 완성했습니다.

### 3. The Polish (검토 및 다듬기)

* AI가 생성한 코드를 Streamlit UI에 통합하고, 사용자 경험(UX)을 고려하여 인터페이스를 다듬었습니다.
* 오류 처리 및 API 연결 부분을 검증하여 안정성을 확보했습니다.

## 🛠️ Tools & Stack

이 프로젝트의 Vibe를 현실로 만들기 위해 사용된 도구들입니다.

* **Brain**: Large Language Models (LLM) - 코드 생성 및 로직 설계
* **Framework**: LangChain, LangGraph - AI 애플리케이션 구조화
* **Interface**: Streamlit - 결과물 시각화
* **Search**: Tavily, FAISS, RankBM25 - 정보 검색 및 처리

## 💭 Conclusion

Vibe Coding을 통해 우리는 **"어떻게 짤 것인가(How)"**보다 **"무엇을 만들 것인가(What)"**에 더 집중할 수 있었습니다. 이 코드는 단순한 텍스트의 나열이 아니라, 개발자의 아이디어와 AI의 실행력이 만나 빚어낸 결과물입니다.
