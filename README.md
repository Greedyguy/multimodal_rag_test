# Multimodal RAG 시스템

여러 개의 문서를 입력받아 이미지로 변환한 뒤, 이미지 임베딩 벡터로 변환하여 저장하고 사용자의 질문에 맞는 이미지를 검색해주는 멀티모달 RAG 시스템입니다.

## 주요 기능

- PDF 파일 업로드 및 이미지 변환
- 이미지 임베딩 생성 및 저장
- Knowledge 단위 데이터 관리
- 텍스트 쿼리 기반 이미지 검색 (단일/대량)
- 검색 결과 평가 및 CSV 다운로드

## 설치 및 실행

### 필수 요구사항

- Python 3.11 이상
- Poetry
- PyTorch (>=2.5.0,<2.7.0)
- ColQwen2.5 모델 지원 환경

### Poetry로 의존성 설치

```bash
# Poetry 설치 (if not installed)
pip install poetry

# 의존성 설치
poetry install

# 가상환경 실행
poetry shell
```

### 애플리케이션 실행

```bash
streamlit run app/main.py
```

## 사용 방법

1. Knowledge 생성 또는 선택
2. PDF/이미지 파일 업로드
3. PDF → 이미지 변환 및 임베딩 생성
4. 단일 쿼리 또는 대량 쿼리(CSV/JSON) 입력
5. 결과 확인 및 다운로드

## 프로젝트 구조

```
multimodal_rag/
├── app/               # 애플리케이션 코드
│   ├── main.py        # Streamlit 진입점
│   ├── models/        # 데이터 모델 및 임베딩 관련 
│   ├── utils/         # 유틸리티 함수 (PDF 변환 등)
│   └── views/         # UI 컴포넌트
├── data/              # 데이터 저장 디렉토리
│   └── knowledge/     # Knowledge별 폴더
├── tests/             # 테스트 코드
├── pyproject.toml     # Poetry 구성 파일
└── README.md          # 이 문서
``` 