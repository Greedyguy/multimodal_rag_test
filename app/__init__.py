"""
Multimodal RAG 시스템 앱 초기화
"""
import os
from pathlib import Path

from app.utils.constants import DATA_DIR, KNOWLEDGE_DIR

# 필요한 디렉토리 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

__version__ = "0.1.0" 