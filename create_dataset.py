from langsmith import Client

client = Client()

# 1. 데이터셋 이름 정의
dataset_name = "UAV Research QA Test"

# 2. 데이터셋 생성 (이미 존재하면 건너뜀)
if client.has_dataset(dataset_name=dataset_name):
    print(f"Dataset '{dataset_name}' already exists.")
else:
    dataset = client.create_dataset(dataset_name=dataset_name, description="UAV RAG 성능 평가용 데이터셋")
    
    # 3. 테스트 데이터 입력 (질문과 예상 답변)
    # 실제 논문 내용을 바탕으로 질문과 정답을 작성해주세요.
    client.create_examples(
        inputs=[
            {"question": "이 논문에서 제안하는 UAV 경로 최적화의 주된 목적은?"},
            {"question": "실험 결과, 제안된 알고리즘은 기존 방식 대비 에너지 효율이 얼마나 향상되었나?"},
            {"question": "시스템 모델에서 가정하고 있는 통신 채널 환경은?"},
        ],
        outputs=[
            {"answer": "UAV의 비행 시간을 최소화하면서 데이터 전송률을 극대화하는 것입니다."},
            {"answer": "기존 방식 대비 약 15%의 에너지 효율 향상을 보였습니다."},
            {"answer": "LoS(Line-of-Sight) 및 NLoS 환경을 모두 고려한 확률적 채널 모델을 가정합니다."},
        ],
        dataset_id=dataset.id,
    )
    print(f"Dataset '{dataset_name}' created with examples.")