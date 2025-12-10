# 1. Base Image: Python 3.11.1을 사용
# slim 버전을 사용하여 이미지 크기를 줄이되, 필요한 도구는 직접 설치
FROM python:3.11.1-slim

# 2. 필수 시스템 패키지 설치
# GitPython을 위한 git, 그리고 C++ 기반 라이브러리 빌드를 위한 build-essential 등을 설치
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사
COPY requirements.txt .

# 5. 라이브러리 설치
# pip를 최신으로 업데이트 후 리스트 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 소스 코드 복사
COPY . .

# 7. 포트 개방 (Streamlit을 사용하므로 8501 포트를 열어둠)
EXPOSE 8501

# 8. 실행 명령어
# 기본적으로 bash를 열어두지만, Streamlit 앱이라면 주석을 해제해서 변경 가능
CMD ["/bin/bash"]
# 만약 바로 앱을 실행하려면 아래와 같이 변경
# CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]

# 9. 빌드 및 실행 방법
# 도커 이미지 빌드: docker build -t kimnote .
# 도커 컨테이너 실행: docker run -it --rm -p 8501:8501 -v ${PWD}:/app kimnote