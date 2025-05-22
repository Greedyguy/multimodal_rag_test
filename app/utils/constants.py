"""
시스템 전역 상수 및 폴더 구조 정의
"""
from pathlib import Path

# 기본 경로 상수
ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"

# Knowledge 폴더 구조
KNOWLEDGE_INFO_FILE = "knowledge_info.json"
PDFS_DIR = "pdfs"
IMAGES_DIR = "images"
EMBEDDINGS_DIR = "embeddings"

# Knowledge ID 형식
KNOWLEDGE_ID_PREFIX = "k_"

# 파일 형식
PDF_EXTENSIONS = [".pdf"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
EMBEDDING_FILE = "embeddings.pt"

# 임베딩 생성 설정
DEFAULT_DPI = 300
MAX_BATCH_SIZE = 4

# UI 관련 상수
DEFAULT_TOP_K = 5

# 파일 이름 패턴 (정규식)
PDF_PAGE_PATTERN = r"(.+)_page_(\d+)\.(png|jpg|jpeg)"

# 엔진 관련 설정
MODEL_NAME = "Metric-AI/ColQwen2.5-3b-multilingual" 