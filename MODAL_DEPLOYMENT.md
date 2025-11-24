# Modal 배포 가이드

## 📋 개요

이 프로젝트는 Modal을 사용하여 GPU 기반 LoRA 학습 및 이미지 생성을 서버리스로 배포합니다.

### 주요 특징
- **서버리스 GPU**: 필요할 때만 GPU 인스턴스 생성
- **자동 스케일링**: 요청량에 따라 자동으로 인스턴스 증감
- **비용 효율적**: 사용한 만큼만 과금
- **멀티 인스턴스**: 동시에 여러 작업 처리 가능

## 🚀 배포 방법

### 1. Modal 설치 및 인증

```bash
# Modal CLI 설치
pip install modal

# Modal 계정 연동
modal token new
```

### 2. 환경 변수 설정

Modal Secret 생성:
```bash
modal secret create lora-secrets \
  AWS_ACCESS_KEY_ID=your-access-key \
  AWS_SECRET_ACCESS_KEY=your-secret-key \
  AWS_S3_BUCKET_NAME=lora-training-data-bucket \
  AWS_S3_REGION=ap-southeast-2
```

### 3. Modal 앱 배포

```bash
# 배포 (FastAPI 서버)
modal deploy modal_app.py

# 배포 완료 후 URL 확인
# 예시: https://yourusername--lora-training-inference-fastapi-app.modal.run
```

### 4. Spring Boot 설정

`application.yml`에 Modal URL 추가:
```yaml
modal:
  app-url: https://yourusername--lora-training-inference-fastapi-app.modal.run
  instances:
    - https://instance1.modal.run
    - https://instance2.modal.run
    - https://instance3.modal.run
  max-instances: 10
  health-check-interval: 30000  # 30초
```

## 📊 인스턴스 관리

### 인스턴스 상태
- `IDLE`: 대기 중 (작업 할당 가능)
- `BUSY`: 작업 처리 중
- `ERROR`: 에러 발생 (자동 복구 시도)
- `OFFLINE`: 오프라인 (헬스 체크 실패)

### 로드 밸런싱 전략
1. **라운드 로빈**: 인스턴스를 순차적으로 할당
2. **최소 부하**: 가장 부하가 적은 인스턴스에 할당
3. **우선순위**: 특정 인스턴스 우선 사용

## 🔧 Modal 함수

### 1. `train_lora()`
- GPU: A10G (학습용)
- 타임아웃: 2시간
- 입력: S3 이미지 URL 리스트
- 출력: 학습된 모델 경로

### 2. `generate_images_modal()`
- GPU: T4 (생성용)
- 타임아웃: 10분
- 입력: 프롬프트, LoRA 모델 URL
- 출력: 생성된 이미지 URL 리스트

### 3. `health_check()`
- GPU: 없음
- 타임아웃: 5초
- 출력: 인스턴스 상태

## 💰 비용 예측

| GPU 타입 | 시간당 비용 | 학습 (250 에포크) | 이미지 생성 (1장) |
|----------|-------------|-------------------|-------------------|
| T4       | $0.60/시간  | ~$1.50            | ~$0.10            |
| A10G     | $1.10/시간  | ~$2.75            | ~$0.18            |
| A100     | $4.00/시간  | ~$10.00           | ~$0.67            |

## 🔍 모니터링

### Modal Dashboard
- https://modal.com/apps
- 실시간 인스턴스 상태 확인
- 로그 및 메트릭 조회
- 비용 추적

### Spring Boot Actuator
```bash
# 인스턴스 상태 조회
GET /actuator/modal/instances

# 헬스 체크
GET /actuator/health
```

## 🐛 트러블슈팅

### 인스턴스가 시작되지 않을 때
1. Modal token 확인: `modal token set`
2. GPU 할당량 확인: Modal Dashboard
3. 로그 확인: `modal logs lora-training-inference`

### 타임아웃 에러
- 학습 시간이 너무 긴 경우: `timeout` 값 증가
- GPU 메모리 부족: 더 강력한 GPU로 변경 (A100)

### 비용이 너무 높을 때
- 인스턴스 수 제한: `max-instances` 감소
- GPU 타입 변경: A10G → T4
- 자동 종료 시간 설정: `idle-timeout` 추가

## 📚 참고 자료

- [Modal 공식 문서](https://modal.com/docs)
- [Modal GPU 가격표](https://modal.com/pricing)
- [FastAPI on Modal](https://modal.com/docs/guide/webhooks)
