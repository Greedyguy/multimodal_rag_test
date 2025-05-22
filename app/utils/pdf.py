"""
PDF 파일 처리 관련 유틸리티 함수
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image
import io

from app.utils.constants import (
    DEFAULT_DPI, PDF_EXTENSIONS, IMAGE_EXTENSIONS,
    PDF_PAGE_PATTERN
)

def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """PDF 페이지 수 반환"""
    with fitz.open(pdf_path) as doc:
        return len(doc)

def convert_pdf_to_images(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    dpi: int = DEFAULT_DPI,
    file_prefix: str = None,
    start_page: int = 0,
    end_page: int = None,
    progress_callback = None
) -> List[Path]:
    """
    PDF 파일을 페이지별 이미지로 변환
    
    Args:
        pdf_path: PDF 파일 경로
        output_dir: 이미지 저장 디렉토리
        dpi: 이미지 해상도 (DPI)
        file_prefix: 파일명 접두사 (없으면 PDF 파일명 사용)
        start_page: 시작 페이지 (0부터 시작)
        end_page: 종료 페이지 (None이면 마지막 페이지까지)
        progress_callback: 진행 상태를 보고하는 콜백 함수
                       함수 형식: callback(current_page, total_pages)
        
    Returns:
        생성된 이미지 파일 경로 리스트
    """
    # 경로 객체 변환
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명 접두사 설정
    if file_prefix is None:
        file_prefix = pdf_path.stem
    
    # dpi 기본값 설정
    if dpi is None:
        dpi = DEFAULT_DPI
    
    # PDF 문서 열기
    doc = fitz.open(pdf_path)
    
    # 페이지 범위 설정
    if end_page is None:
        end_page = len(doc)
    else:
        end_page = min(end_page, len(doc))
    
    total_pages = end_page - start_page
    image_paths = []
    
    # 진행 상태 초기화 (0%)
    if progress_callback:
        progress_callback(0, total_pages)
    
    # 페이지별 이미지 변환
    for i, page_num in enumerate(range(start_page, end_page)):
        page = doc.load_page(page_num)
        
        # 이미지로 렌더링 (픽셀 단위)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        
        # 이미지 파일 경로
        image_path = output_dir / f"{file_prefix}_page_{page_num+1:03d}.png"
        
        # 이미지 저장
        pix.save(str(image_path))
        image_paths.append(image_path)
        
        # 진행 상태 업데이트
        if progress_callback:
            progress_callback(i + 1, total_pages)
    
    doc.close()
    return image_paths

def is_pdf_already_converted(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    file_prefix: str = None
) -> bool:
    """
    PDF가 이미 이미지로 변환되었는지 확인
    
    Args:
        pdf_path: PDF 파일 경로
        output_dir: 이미지 저장 디렉토리
        file_prefix: 파일명 접두사 (없으면 PDF 파일명 사용)
        
    Returns:
        모든 페이지가 이미지로 변환되었으면 True, 아니면 False
    """
    # 경로 객체 변환
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    # 파일명 접두사 설정
    if file_prefix is None:
        file_prefix = pdf_path.stem
    
    # PDF 페이지 수
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
    
    # 변환된 이미지 파일 확인
    for page_num in range(page_count):
        image_path = output_dir / f"{file_prefix}_page_{page_num+1:03d}.png"
        if not image_path.exists():
            return False
    
    return True

def get_conversion_status(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    file_prefix: str = None
) -> Tuple[int, int]:
    """
    PDF 변환 상태 확인
    
    Args:
        pdf_path: PDF 파일 경로
        output_dir: 이미지 저장 디렉토리
        file_prefix: 파일명 접두사 (없으면 PDF 파일명 사용)
        
    Returns:
        (변환된 페이지 수, 총 페이지 수)
    """
    # 경로 객체 변환
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    # 파일명 접두사 설정
    if file_prefix is None:
        file_prefix = pdf_path.stem
    
    # PDF 페이지 수
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    
    # 변환된 이미지 파일 확인
    converted_pages = 0
    for page_num in range(total_pages):
        image_path = output_dir / f"{file_prefix}_page_{page_num+1:03d}.png"
        if image_path.exists():
            converted_pages += 1
    
    return converted_pages, total_pages

def is_valid_pdf(file_path: Union[str, Path]) -> bool:
    """
    유효한 PDF 파일인지 확인
    
    Args:
        file_path: 파일 경로
        
    Returns:
        유효한 PDF 파일이면 True, 아니면 False
    """
    file_path = Path(file_path)
    
    # 파일 확장자 확인
    if file_path.suffix.lower() not in PDF_EXTENSIONS:
        return False
    
    # PDF 파일 열기 가능 여부 확인
    try:
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                return False
            return True
    except Exception:
        return False

def extract_page_info_from_filename(filename: Union[str, Path]) -> Optional[Tuple[str, int]]:
    """
    파일명에서 원본 PDF 이름과 페이지 번호 추출
    
    Args:
        filename: 파일명 또는 경로
        
    Returns:
        (원본 PDF 이름, 페이지 번호) 또는 None
    """
    filename = Path(filename).name
    match = re.match(PDF_PAGE_PATTERN, filename)
    if match:
        pdf_name = match.group(1)
        page_num = int(match.group(2))
        return pdf_name, page_num
    return None 