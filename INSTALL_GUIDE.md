# colpali-engine 설치 가이드

## 맥북에서 임베딩 모듈 설치하기

맥북(Apple Silicon)에서 colpali-engine 모듈을 성공적으로 설치하고 MPS 가속을 활용하기 위한 가이드입니다.

### 필수 패키지 설치

1. **PyTorch 설치 (MPS 지원 버전)**

```bash
# Poetry 환경에서
poetry add torch==2.0.1 torchvision==0.15.2

# 또는 pip 사용 시
pip install torch==2.0.1 torchvision==0.15.2
```

2. **일반 패키지 설치**

```bash
pip install streamlit pillow pandas numpy tqdm
```

3. **colpali-engine 패키지 설치**

```bash
pip install colpali-engine
```

### colpali-engine 설치 실패 시 문제 해결

colpali-engine 설치에 문제가 있는 경우 아래 해결 방법을 시도해보세요:

1. **Transformers 버전 확인**

```bash
# transformers 버전 확인
pip list | grep transformers

# 필요한 경우 특정 버전 설치
pip uninstall -y transformers
pip install transformers==4.30.2
```

2. **PyTorch 재설치**

```bash
pip uninstall -y torch torchvision
pip install torch==2.0.1 torchvision==0.15.2
```

3. **가상환경 초기화**

```bash
# Poetry 환경인 경우
poetry env remove
poetry install

# venv 환경인 경우
python -m venv .venv --clear
source .venv/bin/activate
pip install -r requirements.txt  # 필요한 패키지 다시 설치
```

### MPS 가속 확인

```python
import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"MPS 사용 가능: {torch.backends.mps.is_available()}")
print(f"MPS 빌드 여부: {torch.backends.mps.is_built()}")
```

### 애플리케이션 실행

```bash
# 현재 디렉토리에서 Streamlit 실행
streamlit run app/main.py
```

---

## 문제 해결

### 호환성 문제

colpali-engine이 특정 버전의 torch와 transformers에 의존하는 경우 두 라이브러리 간의 호환성 문제가 발생할 수 있습니다.

- torch 버전: 2.0.1 또는 2.1.0 추천
- transformers 버전: 4.30.x 또는 4.31.x 추천

### CUDA 오류

맥북에서는 CUDA가 아닌 MPS를 사용하므로, CUDA 관련 오류가 발생하면 PyTorch가 MPS를 사용하도록 설정되어 있는지 확인하세요.

### 메모리 오류

임베딩 생성 중 메모리 오류가 발생하면 배치 크기를 줄여보세요 (기본값: 32). UI에서 배치 크기를 조정할 수 있습니다.
