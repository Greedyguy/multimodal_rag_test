"""
Knowledge 관리 모델 및 기능 정의
"""
from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

from app.utils.constants import (
    KNOWLEDGE_DIR, KNOWLEDGE_INFO_FILE,
    PDFS_DIR, IMAGES_DIR, EMBEDDINGS_DIR,
    PDF_EXTENSIONS, DEFAULT_DPI
)
from app.utils.id_generator import generate_knowledge_id, generate_file_id

class KnowledgeInfo(BaseModel):
    """Knowledge 정보 모델"""
    id: str = Field(..., description="고유 ID")
    name: str = Field(..., description="Knowledge 이름")
    description: Optional[str] = Field(None, description="설명")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.now, description="마지막 업데이트 시간")
    
    # 상태 추적 필드
    pdf_count: int = Field(0, description="PDF 파일 수")
    image_count: int = Field(0, description="변환된 이미지 수")
    embedding_count: int = Field(0, description="생성된 임베딩 수")
    
    def model_dump(self) -> Dict[str, Any]:
        """모델을 딕셔너리로 변환"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "pdf_count": self.pdf_count,
            "image_count": self.image_count,
            "embedding_count": self.embedding_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeInfo":
        """딕셔너리에서 모델 생성"""
        # ISO 형식 문자열을 datetime으로 변환
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)

class KnowledgeModel:
    """Knowledge 관련 기능 모델"""
    
    # Knowledge 저장 경로
    BASE_PATH = KNOWLEDGE_DIR
    
    @classmethod
    def get_knowledge_path(cls, knowledge_id: str) -> Path:
        """Knowledge 폴더 경로 반환"""
        return cls.BASE_PATH / knowledge_id
    
    @classmethod
    def get_pdf_path(cls, knowledge_id: str) -> Path:
        """PDF 폴더 경로 반환"""
        return cls.get_knowledge_path(knowledge_id) / PDFS_DIR
    
    @classmethod
    def get_image_path(cls, knowledge_id: str) -> Path:
        """이미지 폴더 경로 반환"""
        return cls.get_knowledge_path(knowledge_id) / IMAGES_DIR
    
    @classmethod
    def get_embedding_path(cls, knowledge_id: str) -> Path:
        """임베딩 폴더 경로 반환"""
        return cls.get_knowledge_path(knowledge_id) / EMBEDDINGS_DIR
    
    @classmethod
    def create_knowledge_directories(cls, knowledge_id: str) -> None:
        """Knowledge 관련 디렉토리 생성"""
        cls.get_pdf_path(knowledge_id).mkdir(parents=True, exist_ok=True)
        cls.get_image_path(knowledge_id).mkdir(parents=True, exist_ok=True)
        cls.get_embedding_path(knowledge_id).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_info_path(cls) -> Path:
        """Knowledge 정보 파일 경로 반환"""
        path = cls.BASE_PATH / KNOWLEDGE_INFO_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def create_knowledge(cls, name: str, description: str = None) -> KnowledgeInfo:
        """새 Knowledge 생성"""
        # 고유 ID 생성
        knowledge_id = generate_knowledge_id()
        
        # Knowledge 정보 생성
        knowledge = KnowledgeInfo(
            id=knowledge_id,
            name=name,
            description=description
        )
        
        # 디렉토리 생성
        cls.create_knowledge_directories(knowledge_id)
        
        # Knowledge 정보 저장
        cls.save_knowledge_info(knowledge)
        
        return knowledge
    
    @classmethod
    def save_knowledge_info(cls, knowledge: KnowledgeInfo) -> bool:
        """Knowledge 정보 저장"""
        try:
            # 모든 Knowledge 목록 로드
            knowledge_list = cls.list_knowledge_infos()
            
            # 기존 knowledge 확인 후 업데이트 또는 추가
            updated = False
            for i, k in enumerate(knowledge_list):
                if k["id"] == knowledge.id:
                    knowledge_list[i] = knowledge.model_dump()
                    updated = True
                    break
            
            if not updated:
                knowledge_list.append(knowledge.model_dump())
            
            # 파일 저장
            cls.get_info_path().parent.mkdir(parents=True, exist_ok=True)
            with open(cls.get_info_path(), 'w', encoding='utf-8') as f:
                json.dump(knowledge_list, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Knowledge 정보 저장 중 오류 발생: {e}")
            return False
    
    @classmethod
    def list_knowledge_infos(cls) -> List[Dict[str, Any]]:
        """모든 Knowledge 정보 목록 반환"""
        try:
            if not cls.get_info_path().exists():
                return []
            
            with open(cls.get_info_path(), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Knowledge 정보 로드 중 오류 발생: {e}")
            return []
    
    @classmethod
    def get_knowledge_infos(cls) -> List[KnowledgeInfo]:
        """모든 Knowledge 객체 목록 반환"""
        knowledge_dicts = cls.list_knowledge_infos()
        return [KnowledgeInfo.from_dict(k) for k in knowledge_dicts]
    
    @classmethod
    def get_knowledge_info(cls, knowledge_id: str) -> Optional[KnowledgeInfo]:
        """특정 Knowledge 정보 반환"""
        for knowledge_dict in cls.list_knowledge_infos():
            if knowledge_dict["id"] == knowledge_id:
                return KnowledgeInfo.from_dict(knowledge_dict)
        return None
    
    @classmethod
    def update_knowledge_status(cls, knowledge_id: str, 
                                pdf_count: Optional[int] = None,
                                image_count: Optional[int] = None,
                                embedding_count: Optional[int] = None) -> bool:
        """Knowledge 상태 업데이트"""
        knowledge = cls.get_knowledge_info(knowledge_id)
        if knowledge is None:
            return False
        
        # 상태 업데이트
        if pdf_count is not None:
            knowledge.pdf_count = pdf_count
        
        if image_count is not None:
            knowledge.image_count = image_count
        
        if embedding_count is not None:
            knowledge.embedding_count = embedding_count
        
        # 업데이트 시간 갱신
        knowledge.updated_at = datetime.now()
        
        # 저장
        return cls.save_knowledge_info(knowledge)
    
    @classmethod
    def delete_knowledge(cls, knowledge_id: str) -> bool:
        """Knowledge 삭제"""
        try:
            # 모든 Knowledge 목록 로드
            knowledge_list = cls.list_knowledge_infos()
            
            # 삭제할 knowledge 찾기
            knowledge_list = [k for k in knowledge_list if k["id"] != knowledge_id]
            
            # 파일 저장
            with open(cls.get_info_path(), 'w', encoding='utf-8') as f:
                json.dump(knowledge_list, f, ensure_ascii=False, indent=2)
            
            # TODO: 폴더와 파일 삭제 로직 추가 (선택사항)
            
            return True
        except Exception as e:
            print(f"Knowledge 삭제 중 오류 발생: {e}")
            return False 
            
    @classmethod
    def save_pdf_file(cls, knowledge_id: str, pdf_file_path: Union[str, Path], 
                      new_filename: Optional[str] = None) -> Optional[Path]:
        """
        PDF 파일을 Knowledge 디렉토리에 저장
        
        Args:
            knowledge_id: Knowledge ID
            pdf_file_path: 저장할 PDF 파일 경로
            new_filename: 새 파일명 (None이면 고유 ID 기반 생성)
            
        Returns:
            저장된 파일 경로 또는 오류 시 None
        """
        try:
            # 경로 객체 변환
            pdf_file_path = Path(pdf_file_path)
            
            # 유효한 PDF 파일 확인
            if pdf_file_path.suffix.lower() not in PDF_EXTENSIONS:
                print(f"유효하지 않은 PDF 파일 형식: {pdf_file_path}")
                return None
            
            # 파일명 생성
            if new_filename is None:
                file_id = generate_file_id(prefix="pdf")
                new_filename = f"{file_id}{pdf_file_path.suffix}"
            
            # 저장 경로
            pdf_dir = cls.get_pdf_path(knowledge_id)
            pdf_dir.mkdir(parents=True, exist_ok=True)
            target_path = pdf_dir / new_filename
            
            # 파일 복사
            shutil.copy2(pdf_file_path, target_path)
            
            # PDF 카운트 업데이트
            knowledge = cls.get_knowledge_info(knowledge_id)
            if knowledge:
                pdf_count = len(list(pdf_dir.glob("*.*")))
                cls.update_knowledge_status(knowledge_id, pdf_count=pdf_count)
            
            return target_path
        except Exception as e:
            print(f"PDF 파일 저장 중 오류 발생: {e}")
            return None
    
    @classmethod
    def list_pdf_files(cls, knowledge_id: str) -> List[Path]:
        """
        Knowledge에 저장된 PDF 파일 목록 반환
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            PDF 파일 경로 리스트
        """
        pdf_dir = cls.get_pdf_path(knowledge_id)
        if not pdf_dir.exists():
            return []
        
        # PDF 파일만 필터링
        pdf_files = []
        for ext in PDF_EXTENSIONS:
            pdf_files.extend(pdf_dir.glob(f"*{ext}"))
        
        return sorted(pdf_files)
    
    @classmethod
    def convert_pdf_to_images(cls, knowledge_id: str, pdf_file: Union[str, Path], 
                              dpi: int = None, progress_callback=None, 
                              file_index: int = 0, total_files: int = 1) -> List[Path]:
        """
        PDF 파일을 이미지로 변환하여 Knowledge의 이미지 디렉토리에 저장
        
        Args:
            knowledge_id: Knowledge ID
            pdf_file: PDF 파일 경로
            dpi: 이미지 해상도 (None이면 기본값 사용)
            progress_callback: 진행 상태를 보고하는 콜백 함수
            file_index: 전체 PDF 중 현재 처리 중인 파일 인덱스
            total_files: 전체 PDF 파일 수
            
        Returns:
            생성된 이미지 파일 경로 리스트
        """
        try:
            from app.utils.pdf import convert_pdf_to_images as convert_pdf
            from app.utils.pdf import get_pdf_page_count
            
            # 경로 객체 변환
            pdf_file = Path(pdf_file)
            
            # 파일명에서 접두사 추출 (확장자 제외)
            file_prefix = pdf_file.stem
            
            # 이미지 저장 디렉토리
            output_dir = cls.get_image_path(knowledge_id)
            
            # dpi가 None인 경우 기본값 사용
            if dpi is None:
                dpi = DEFAULT_DPI
            
            # PDF 페이지 수 확인
            total_pages = get_pdf_page_count(pdf_file)
            
            # 페이지별 진행 상태를 보고하는 래퍼 함수
            def page_progress_callback(page_num, total_pages):
                if progress_callback:
                    progress_callback(pdf_file.name, page_num, total_pages, file_index, total_files)
            
            # PDF를 이미지로 변환
            image_paths = convert_pdf(
                pdf_path=pdf_file,
                output_dir=output_dir,
                dpi=dpi,
                file_prefix=file_prefix,
                progress_callback=page_progress_callback
            )
            
            # 이미지 개수 업데이트
            if image_paths:
                image_count = len(list(output_dir.glob("*.png"))) + len(list(output_dir.glob("*.jpg"))) + len(list(output_dir.glob("*.jpeg")))
                cls.update_knowledge_status(knowledge_id, image_count=image_count)
            
            return image_paths
        except Exception as e:
            print(f"PDF 이미지 변환 중 오류 발생: {e}")
            return []
    
    @classmethod
    def convert_all_pdfs(cls, knowledge_id: str, dpi: int = None, progress_callback=None) -> Dict[str, List[Path]]:
        """
        Knowledge의 모든 PDF 파일을 이미지로 변환
        
        Args:
            knowledge_id: Knowledge ID
            dpi: 이미지 해상도 (None이면 기본값 사용)
            progress_callback: 진행 상태를 보고하는 콜백 함수
                            함수 형식: callback(current_file, current_page, total_pages, file_index, total_files)
            
        Returns:
            PDF 파일별 생성된 이미지 경로 딕셔너리
        """
        from app.utils.pdf import is_pdf_already_converted, get_pdf_page_count
        
        # PDF 파일 목록 조회
        pdf_files = cls.list_pdf_files(knowledge_id)
        total_files = len(pdf_files)
        
        # 이미지 저장 디렉토리
        output_dir = cls.get_image_path(knowledge_id)
        
        # 변환 결과
        results = {}
        
        # 모든 PDF 변환
        for file_index, pdf_file in enumerate(pdf_files):
            # 진행 상태 보고 (시작)
            if progress_callback:
                progress_callback(pdf_file.name, 0, 1, file_index, total_files)
            
            # 이미 변환된 PDF는 건너뛰기
            if is_pdf_already_converted(pdf_file, output_dir, file_prefix=pdf_file.stem):
                # 기존 이미지 파일 경로 수집
                existing_images = sorted(output_dir.glob(f"{pdf_file.stem}_page_*.png"))
                results[str(pdf_file)] = existing_images
                
                # 진행 상태 보고 (완료)
                if progress_callback:
                    pages = len(existing_images)
                    progress_callback(pdf_file.name, pages, pages, file_index, total_files)
                
                continue
            
            # PDF를 이미지로 변환
            image_paths = cls.convert_pdf_to_images(knowledge_id, pdf_file, dpi=dpi, progress_callback=progress_callback,
                                                  file_index=file_index, total_files=total_files)
            results[str(pdf_file)] = image_paths
        
        return results
    
    @classmethod
    def list_images(cls, knowledge_id: str) -> List[Path]:
        """
        Knowledge에 저장된 이미지 파일 목록 반환
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            이미지 파일 경로 리스트
        """
        from app.utils.constants import IMAGE_EXTENSIONS
        
        image_dir = cls.get_image_path(knowledge_id)
        if not image_dir.exists():
            return []
        
        # 이미지 파일만 필터링
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(image_dir.glob(f"*{ext}"))
        
        return sorted(image_files)
    
    @classmethod
    def generate_embeddings(cls, knowledge_id: str, 
                           batch_size: int = None, 
                           progress_callback = None,
                           max_image_size: int = 1024,
                           save_interval: int = 50,
                           low_memory_mode: bool = True) -> bool:
        """
        Knowledge에 저장된 이미지의 임베딩을 생성하여 저장
        
        Args:
            knowledge_id: Knowledge ID
            batch_size: 배치 처리 크기 (None이면 기본값 사용)
            progress_callback: 진행 상태를 보고하는 콜백 함수
                           함수 형식: callback(current_index, total_count, status_message)
            max_image_size: 이미지 최대 크기 (px)
            save_interval: 중간 저장할 이미지 간격
            low_memory_mode: 저메모리 모드 활성화 여부
            
        Returns:
            성공 여부
        """
        try:
            # 환경 변수 설정 (lzma 및 기타 오류 방지)
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["PYTHONIOENCODING"] = "utf-8"
            
            # 이미지 목록 조회
            image_files = cls.list_images(knowledge_id)
            if not image_files:
                if progress_callback:
                    progress_callback(0, 1, "임베딩을 생성할 이미지가 없습니다.")
                print(f"임베딩을 생성할 이미지가 없습니다. Knowledge ID: {knowledge_id}")
                return False
            
            # 임베딩 모듈 초기화
            try:
                # importlib로 모듈 존재 여부 확인
                import importlib.util
                colpali_spec = importlib.util.find_spec("colpali_engine")
                
                if colpali_spec is None:
                    if progress_callback:
                        progress_callback(0, 1, "colpali 모듈을 찾을 수 없습니다.")
                    print("colpali_engine 모듈을 찾을 수 없습니다. 모듈 설치가 필요합니다.")
                    print("설치 방법: pip install colpali-engine")
                    print("자세한 내용은 INSTALL_GUIDE.md 파일을 참조하세요.")
                    return False
                
                # 모듈이 존재하는 경우에만 임포트
                try:
                    from app.models.embedding import EmbeddingManager
                    embedding_manager = EmbeddingManager()
                except ImportError as e:
                    if progress_callback:
                        progress_callback(0, 1, f"임베딩 모듈 로드 오류: {str(e)}")
                    print(f"EmbeddingManager 모듈 로드 중 오류: {e}")
                    return False
                except Exception as e:
                    if progress_callback:
                        progress_callback(0, 1, f"임베딩 매니저 초기화 오류: {str(e)}")
                    print(f"EmbeddingManager 초기화 중 오류: {e}")
                    return False
            except ImportError as e:
                if "colpali_engine" in str(e):
                    if progress_callback:
                        progress_callback(0, 1, "colpali 모듈을 찾을 수 없습니다.")
                    print(f"colpali_engine 모듈 오류: {e}")
                    print("설치 방법: pip install colpali-engine")
                    return False
                else:
                    if progress_callback:
                        progress_callback(0, 1, f"모듈 오류: {str(e)}")
                    print(f"모듈 임포트 오류: {e}")
                    return False
            
            # 임베딩 저장 경로
            embedding_dir = cls.get_embedding_path(knowledge_id)
            embedding_file = embedding_dir / "embeddings.pt"  # EMBEDDING_FILE 상수값 사용
            
            # 이미 처리된 이미지 정보 (기존에 임베딩이 있는 경우)
            existing_embeddings = None
            processed_image_paths = set()
            
            # 기존 임베딩 파일 확인 (이어서 처리하기 위함)
            if embedding_file.exists():
                try:
                    print(f"기존 임베딩 파일 발견: {embedding_file}")
                    if progress_callback:
                        progress_callback(0, len(image_files), "기존 임베딩 로드 중...")
                        
                    # 기존 임베딩 로드
                    existing_embeddings = embedding_manager.load_embeddings(embedding_dir)
                    
                    if existing_embeddings and "file_names" in existing_embeddings:
                        # 이미 처리된 이미지 경로 집합 생성
                        processed_image_paths = set(existing_embeddings["file_names"])
                        
                        # 이미 처리된 이미지 수
                        processed_count = len(processed_image_paths)
                        print(f"기존 임베딩 로드 완료: {processed_count}개 이미지 처리됨")
                        
                        if progress_callback:
                            progress_callback(0, len(image_files), 
                                             f"기존 임베딩 로드 완료: {processed_count}개 이미지 이미 처리됨")
                            
                        # 모든 이미지가 이미 처리된 경우
                        if len(processed_image_paths) >= len(image_files):
                            print("모든 이미지가 이미 처리되었습니다.")
                            if progress_callback:
                                progress_callback(len(image_files), len(image_files), 
                                                "모든 이미지가 이미 처리되었습니다.")
                            return True
                    else:
                        print("기존 임베딩 파일에 유효한 데이터가 없습니다. 새로 처리합니다.")
                except Exception as e:
                    print(f"기존 임베딩 로드 중 오류 발생: {e}")
                    print("기존 임베딩 무시하고 새로 처리합니다.")
                    existing_embeddings = None
                    processed_image_paths = set()
            
            # 처리할 이미지 필터링 (이미 처리된 이미지 제외)
            if processed_image_paths:
                original_count = len(image_files)
                image_files = [img for img in image_files if str(img) not in processed_image_paths]
                skipped_count = original_count - len(image_files)
                print(f"이미 처리된 {skipped_count}개 이미지 건너뜀, 남은 이미지: {len(image_files)}개")
            
            # 처리할 이미지가 없는 경우
            if not image_files:
                print("모든 이미지가 이미 처리되었습니다.")
                if progress_callback:
                    progress_callback(1, 1, "모든 이미지가 이미 처리되었습니다.")
                return True
            
            if progress_callback:
                progress_callback(0, len(image_files), f"총 {len(image_files)}개 이미지 임베딩 생성 준비 중...")
            
            # 임베딩 생성 상태 업데이트 함수
            def update_embedding_progress(current, total, message):
                if progress_callback:
                    progress_callback(current, total, message)
            
            # 임베딩 생성
            try:
                # 설정 정보 출력
                print(f"임베딩 생성 설정: max_image_size={max_image_size}, save_interval={save_interval}, low_memory_mode={low_memory_mode}")
                
                # 새 이미지 임베딩 생성
                new_embeddings = embedding_manager.process_images(
                    image_paths=image_files,
                    batch_size=batch_size,
                    progress_callback=update_embedding_progress,
                    max_image_size=max_image_size,
                    save_interval=save_interval,
                    low_memory_mode=low_memory_mode
                )
                
                if not new_embeddings or "embeddings" not in new_embeddings or not new_embeddings["embeddings"]:
                    if progress_callback:
                        progress_callback(0, 1, "임베딩 생성 결과가 비어있습니다.")
                    print("임베딩 생성 결과가 비어있습니다.")
                    
                    # 기존 임베딩이 있으면 그대로 유지
                    if existing_embeddings and "embeddings" in existing_embeddings and existing_embeddings["embeddings"]:
                        if progress_callback:
                            progress_callback(1, 1, "기존 임베딩 유지합니다.")
                        return True
                    
                    return False
                
                # 기존 임베딩과 새 임베딩 병합
                if existing_embeddings and "embeddings" in existing_embeddings and existing_embeddings["embeddings"]:
                    if progress_callback:
                        progress_callback(len(image_files), len(image_files), "기존 임베딩과 새 임베딩 병합 중...")
                    
                    # 임베딩 병합
                    merged_embeddings = {
                        "file_names": existing_embeddings["file_names"] + new_embeddings["file_names"],
                        "page_nums": existing_embeddings["page_nums"] + new_embeddings["page_nums"],
                        "embeddings": existing_embeddings["embeddings"] + new_embeddings["embeddings"],
                        "doc_ids": existing_embeddings["doc_ids"] + new_embeddings["doc_ids"]
                    }
                    
                    embeddings_data = merged_embeddings
                    print(f"임베딩 병합 완료: {len(existing_embeddings['embeddings'])}개 기존 + {len(new_embeddings['embeddings'])}개 새로운 = {len(merged_embeddings['embeddings'])}개 총 임베딩")
                else:
                    # 새 임베딩만 사용
                    embeddings_data = new_embeddings
                
                if progress_callback:
                    progress_callback(len(image_files), len(image_files), "임베딩 저장 중...")
                
                # 임베딩 저장
                success = embedding_manager.save_embeddings(embeddings_data, embedding_dir)
                
                # 임베딩 개수 업데이트
                if success:
                    embedding_count = len(embeddings_data["embeddings"])
                    cls.update_knowledge_status(knowledge_id, embedding_count=embedding_count)
                    if progress_callback:
                        progress_callback(len(image_files), len(image_files), f"{embedding_count}개 임베딩 저장 완료")
                else:
                    if progress_callback:
                        progress_callback(len(image_files), len(image_files), "임베딩 저장 실패")
                
                return success
            except Exception as e:
                if progress_callback:
                    progress_callback(0, 1, f"임베딩 생성 오류: {str(e)}")
                print(f"임베딩 생성 과정 중 오류: {e}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            if progress_callback:
                progress_callback(0, 1, f"임베딩 생성 기능 오류: {str(e)}")
            print(f"임베딩 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @classmethod
    def load_embeddings(cls, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Knowledge의 임베딩 로드
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            임베딩 데이터 딕셔너리 또는 오류 시 None
        """
        try:
            from app.models.embedding import EmbeddingManager
            
            # 임베딩 매니저 초기화
            embedding_manager = EmbeddingManager()
            
            # 임베딩 저장 경로
            embedding_dir = cls.get_embedding_path(knowledge_id)
            
            # 임베딩 로드
            embeddings_data = embedding_manager.load_embeddings(embedding_dir)
            
            return embeddings_data
        except Exception as e:
            print(f"임베딩 로드 중 오류 발생: {e}")
            return None
    
    @classmethod
    def get_knowledge_statistics(cls, knowledge_id: str) -> Dict[str, Any]:
        """
        Knowledge의 통계 정보 반환
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            통계 정보 딕셔너리
        """
        # 기본 통계 정보
        stats = {
            "pdf_count": 0,
            "image_count": 0,
            "embedding_count": 0,
            "knowledge_info": None
        }
        
        try:
            # Knowledge 정보 조회
            knowledge = cls.get_knowledge_info(knowledge_id)
            if knowledge:
                stats["knowledge_info"] = knowledge.model_dump()
                stats["pdf_count"] = knowledge.pdf_count
                stats["image_count"] = knowledge.image_count
                stats["embedding_count"] = knowledge.embedding_count
            
            # 파일 기반 통계 (실제 파일 수 기준)
            pdf_dir = cls.get_pdf_path(knowledge_id)
            if pdf_dir.exists():
                pdf_files = cls.list_pdf_files(knowledge_id)
                stats["actual_pdf_count"] = len(pdf_files)
            
            image_dir = cls.get_image_path(knowledge_id)
            if image_dir.exists():
                image_files = cls.list_images(knowledge_id)
                stats["actual_image_count"] = len(image_files)
            
            # 임베딩 통계
            from app.models.embedding import EmbeddingManager
            embedding_manager = EmbeddingManager()
            embedding_dir = cls.get_embedding_path(knowledge_id)
            if embedding_dir.exists():
                embedding_stats = embedding_manager.get_embedding_stats(embedding_dir)
                stats.update(embedding_stats)
            
            return stats
        except Exception as e:
            print(f"통계 정보 수집 중 오류 발생: {e}")
            return stats
            
    @classmethod
    def search_knowledge_by_name(cls, name_query: str) -> List[KnowledgeInfo]:
        """
        이름으로 Knowledge 검색
        
        Args:
            name_query: 검색 쿼리
            
        Returns:
            검색 결과 Knowledge 목록
        """
        knowledge_list = cls.get_knowledge_infos()
        
        # 대소문자 구분 없이 검색
        name_query = name_query.lower()
        
        # 이름에 쿼리가 포함된 Knowledge 필터링
        filtered = [k for k in knowledge_list if name_query in k.name.lower()]
        
        return filtered
    
    @classmethod
    def filter_knowledge_by_criteria(cls, 
                                  min_pdfs: Optional[int] = None,
                                  min_images: Optional[int] = None,
                                  min_embeddings: Optional[int] = None,
                                  has_embeddings: Optional[bool] = None,
                                  creation_after: Optional[datetime] = None,
                                  creation_before: Optional[datetime] = None,
                                  updated_after: Optional[datetime] = None,
                                  updated_before: Optional[datetime] = None) -> List[KnowledgeInfo]:
        """
        다양한 기준으로 Knowledge 필터링
        
        Args:
            min_pdfs: 최소 PDF 파일 수
            min_images: 최소 이미지 파일 수
            min_embeddings: 최소 임베딩 수
            has_embeddings: 임베딩 존재 여부
            creation_after: 이후에 생성된 Knowledge 필터링
            creation_before: 이전에 생성된 Knowledge 필터링
            updated_after: 이후에 업데이트된 Knowledge 필터링
            updated_before: 이전에 업데이트된 Knowledge 필터링
            
        Returns:
            필터링된 Knowledge 목록
        """
        knowledge_list = cls.get_knowledge_infos()
        filtered = []
        
        for knowledge in knowledge_list:
            # 모든 필터 조건 검사
            if min_pdfs is not None and knowledge.pdf_count < min_pdfs:
                continue
            
            if min_images is not None and knowledge.image_count < min_images:
                continue
            
            if min_embeddings is not None and knowledge.embedding_count < min_embeddings:
                continue
            
            if has_embeddings is not None:
                has_emb = knowledge.embedding_count > 0
                if has_emb != has_embeddings:
                    continue
            
            if creation_after is not None and knowledge.created_at < creation_after:
                continue
            
            if creation_before is not None and knowledge.created_at > creation_before:
                continue
            
            if updated_after is not None and knowledge.updated_at < updated_after:
                continue
            
            if updated_before is not None and knowledge.updated_at > updated_before:
                continue
            
            # 모든 조건을 통과한 Knowledge 추가
            filtered.append(knowledge)
        
        return filtered
    
    @classmethod
    def search_pdf_files(cls, knowledge_id: str, keyword: str) -> List[Path]:
        """
        PDF 파일명으로 검색
        
        Args:
            knowledge_id: Knowledge ID
            keyword: 검색 키워드
            
        Returns:
            검색 결과 파일 경로 리스트
        """
        pdf_files = cls.list_pdf_files(knowledge_id)
        
        # 대소문자 구분 없이 검색
        keyword = keyword.lower()
        
        # 파일명에 키워드가 포함된 파일 필터링
        filtered = [pdf for pdf in pdf_files if keyword in pdf.stem.lower()]
        
        return filtered
    
    @classmethod
    def search_images(cls, knowledge_id: str, keyword: str) -> List[Path]:
        """
        이미지 파일명으로 검색
        
        Args:
            knowledge_id: Knowledge ID
            keyword: 검색 키워드
            
        Returns:
            검색 결과 파일 경로 리스트
        """
        image_files = cls.list_images(knowledge_id)
        
        # 대소문자 구분 없이 검색
        keyword = keyword.lower()
        
        # 파일명에 키워드가 포함된 파일 필터링
        filtered = [img for img in image_files if keyword in img.stem.lower()]
        
        return filtered
        
    @classmethod
    def get_knowledge_by_file_count(cls, count_type: str, 
                                 min_count: Optional[int] = None, 
                                 max_count: Optional[int] = None) -> List[KnowledgeInfo]:
        """
        파일 수로 Knowledge 검색
        
        Args:
            count_type: 파일 유형 ("pdf", "image", "embedding")
            min_count: 최소 파일 수
            max_count: 최대 파일 수
            
        Returns:
            필터링된 Knowledge 목록
        """
        knowledge_list = cls.get_knowledge_infos()
        filtered = []
        
        for knowledge in knowledge_list:
            # 파일 유형별 카운트 가져오기
            if count_type == "pdf":
                count = knowledge.pdf_count
            elif count_type == "image":
                count = knowledge.image_count
            elif count_type == "embedding":
                count = knowledge.embedding_count
            else:
                # 지원하지 않는 파일 유형
                return []
            
            # 최소/최대 카운트 필터링
            if min_count is not None and count < min_count:
                continue
            
            if max_count is not None and count > max_count:
                continue
            
            filtered.append(knowledge)
        
        return filtered
        
    @classmethod
    def sort_knowledge_by_field(cls, field: str, ascending: bool = True) -> List[KnowledgeInfo]:
        """
        특정 필드로 Knowledge 정렬
        
        Args:
            field: 정렬 기준 필드 (name, created_at, updated_at, pdf_count, image_count, embedding_count)
            ascending: 오름차순 정렬 여부
            
        Returns:
            정렬된 Knowledge 목록
        """
        knowledge_list = cls.get_knowledge_infos()
        
        # 지원하는 필드 목록
        valid_fields = ["name", "created_at", "updated_at", "pdf_count", "image_count", "embedding_count"]
        
        if field not in valid_fields:
            print(f"지원하지 않는 정렬 필드: {field}. 기본 이름순 정렬을 수행합니다.")
            field = "name"
        
        # 필드 기준으로 정렬
        sorted_list = sorted(knowledge_list, key=lambda k: getattr(k, field), reverse=not ascending)
        
        return sorted_list 