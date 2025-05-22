"""
고유 ID 생성 유틸리티
"""
import time
import uuid
from datetime import datetime
import random
import string
from typing import Dict, List, Optional

from app.utils.constants import KNOWLEDGE_ID_PREFIX

def generate_knowledge_id() -> str:
    """
    Knowledge ID 생성
    
    형식: k_<timestamp>
    예: k_1679012345
    
    Returns:
        고유 Knowledge ID
    """
    timestamp = int(datetime.now().timestamp())
    return f"{KNOWLEDGE_ID_PREFIX}{timestamp}"

def generate_file_id(prefix: str = "f") -> str:
    """
    파일 ID 생성
    
    형식: <prefix>_<timestamp>_<random>
    예: f_1679012345_a1b2c3
    
    Args:
        prefix: ID 접두사 (기본값: "f")
        
    Returns:
        고유 파일 ID
    """
    timestamp = int(time.time())
    random_str = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"{prefix}_{timestamp}_{random_str}"

def generate_uuid() -> str:
    """
    UUID 기반 ID 생성
    
    Returns:
        UUID 문자열
    """
    return str(uuid.uuid4())

def generate_embed_id(file_name: str, page_num: int) -> str:
    """
    임베딩 ID 생성
    
    형식: <file_name>_p<page_num>
    예: document1_p5
    
    Args:
        file_name: 파일명
        page_num: 페이지 번호
        
    Returns:
        임베딩 ID
    """
    # 파일 확장자 제거
    base_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
    # 경로 제거
    base_name = base_name.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    return f"{base_name}_p{page_num}"

def generate_session_id() -> str:
    """
    세션 ID 생성
    
    형식: s_<timestamp>_<random>
    예: s_1679012345_a1b2
    
    Returns:
        세션 ID
    """
    timestamp = int(time.time())
    random_str = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    return f"s_{timestamp}_{random_str}" 