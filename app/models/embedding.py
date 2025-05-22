"""
임베딩 생성 및 관리 모듈
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import sys
import tempfile
import gc

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from app.utils.constants import (
    MODEL_NAME, EMBEDDING_FILE, MAX_BATCH_SIZE
)

# lzma 모듈 문제 및 dynamo 이슈 해결을 위한 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Torch 컴파일 비활성화 (Apple Silicon 호환성 개선)
try:
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 64
except:
    pass

class EmbeddingManager:
    """
    이미지 임베딩 생성 및 관리 클래스
    
    ColQwen2.5 모델을 사용하여 이미지 임베딩을 생성하고 관리합니다.
    """
    
    # 싱글톤 인스턴스
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.processor = None
            
            # MPS (Apple Silicon), CUDA, CPU 순으로 기기 선택
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("MPS (Apple Silicon GPU) 가속 사용")
                cls._instance.device = torch.device("mps")
            elif torch.cuda.is_available():
                print("CUDA GPU 가속 사용")
                cls._instance.device = torch.device("cuda:0")
            else:
                print("CPU 사용")
                cls._instance.device = torch.device("cpu")
                
        return cls._instance
    
    def __init__(self, model=None, processor=None):
        """
        EmbeddingManager 초기화
        
        Args:
            model: 사용할 모델 (기본값: None, 필요시 로드)
            processor: 사용할 프로세서 (기본값: None, 필요시 로드)
        """
        # 싱글톤으로 초기화된 경우에만 값을 업데이트
        if model is not None:
            self.model = model
        if processor is not None:
            self.processor = processor
    
    def load_model(self):
        """
        ColQwen2.5 모델 및 프로세서 로드
        """
        try:
            # 모듈 존재 여부 확인
            import importlib.util
            colpali_spec = importlib.util.find_spec("colpali_engine")
            
            if colpali_spec is None:
                print("임베딩 모듈(colpali_engine)이 설치되어 있지 않습니다.")
                print("설치 방법: pip install colpali-engine")
                return False
            
            # _lzma 모듈 오류 방지를 위한 임시 조치
            if 'lzma' in sys.modules:
                # 이미 로드된 경우 삭제
                del sys.modules['lzma']
            
            # 필요한 모듈만 선택적 임포트 시도
            try:
                # colpali_engine 모듈 임포트 시도
                from colpali_engine import ColQwen2_5, ColQwen2_5_Processor
            except ImportError as e:
                if "_lzma" in str(e) or "lzma" in str(e):
                    print("_lzma 모듈 오류 발생: MacOS에서 알려진 이슈입니다")
                    print("대체 임베딩 방식으로 전환이 필요합니다.")
                    return False
                elif "torch._C._dynamo.eval_frame" in str(e):
                    print("Torch dynamo 관련 오류 발생: 호환성 문제가 있습니다")
                    return False
                else:
                    print(f"colpali_engine 모듈을 임포트할 수 없습니다: {e}")
                    print("설치 방법: pip install colpali-engine")
                    return False
            
            # 아직 모델이 로드되지 않은 경우에만 로드
            if self.model is None:
                print(f"모델 로딩 중... ({MODEL_NAME})")
                try:
                    self.model = ColQwen2_5.from_pretrained(
                        MODEL_NAME,
                        torch_dtype=torch.float16 if self.device.type == "mps" else torch.bfloat16,
                        device_map=self.device,
                    ).eval()
                except Exception as e:
                    print(f"모델 로드 중 오류: {e}")
                    # CPU로 폴백
                    print("CPU로 전환하여 다시 시도합니다...")
                    self.device = torch.device("cpu")
                    self.model = ColQwen2_5.from_pretrained(
                        MODEL_NAME,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                    ).eval()
            
            if self.processor is None:
                print(f"프로세서 로딩 중... ({MODEL_NAME})")
                self.processor = ColQwen2_5_Processor.from_pretrained(
                    MODEL_NAME
                )
            
            return True
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}")
            print("상세 오류 정보:")
            import traceback
            traceback.print_exc()
            return False
    
    def resize_image_if_needed(self, img: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        이미지 크기가 너무 크면 리사이즈
        Args:
            img: PIL 이미지
            max_size: 최대 폭/높이 (기본값 1024px, None이면 리사이즈 안함)
        Returns:
            리사이즈된 이미지 또는 원본 이미지
        """
        if max_size is None:
            return img  # 무조건 원본 반환
        width, height = img.size
        if width <= max_size and height <= max_size:
            return img
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        print(f"이미지 리사이즈: {width}x{height} -> {new_width}x{new_height} (max_size={max_size})")
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def get_image_embedding(self, image_path: Union[str, Path], max_image_size: int = 1024) -> Optional[torch.Tensor]:
        """
        이미지 임베딩 생성
        Args:
            image_path: 이미지 파일 경로
            max_image_size: 최대 이미지 크기 (px, None이면 리사이즈 안함)
        Returns:
            이미지 임베딩 텐서 또는 오류 시 None
        """
        try:
            if self.model is None or self.processor is None:
                if not self.load_model():
                    return None
            img = Image.open(image_path).convert("RGB")
            img = self.resize_image_if_needed(img, max_size=max_image_size)
            processed_img = self.processor.process_images([img])
            processed_img = {k: v.to(self.device) for k, v in processed_img.items()}
            with torch.no_grad():
                embedding = self.model(**processed_img)
            unbinded_embeddings = list(torch.unbind(embedding.to("cpu").detach()))
            if unbinded_embeddings:
                return unbinded_embeddings
            else:
                return None
        except Exception as e:
            print(f"이미지 임베딩 생성 중 오류 발생: {e}")
            return None
    
    def clear_memory(self):
        """
        메모리 정리 함수
        GPU/MPS 캐시 및 가비지 컬렉션을 수행합니다.
        """
        import gc
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # MPS 메모리 정리 (Apple Silicon)
        if hasattr(torch, 'mps'):
            try:
                torch.mps.empty_cache()
            except:
                pass  # M1/M2 맥북에서만 작동하는 함수
        
        # 가비지 컬렉션 강제 실행
        gc.collect()

    def process_images(self, 
                      image_paths: List[Union[str, Path]],
                      batch_size: int = MAX_BATCH_SIZE,
                      progress_callback = None,
                      save_interval: int = 50,  # 50개 이미지마다 중간 저장
                      temp_output_dir: Optional[Union[str, Path]] = None,
                      max_image_size: int = 1024,
                      low_memory_mode: bool = True) -> Dict[str, Any]:
        """
        여러 이미지의 임베딩 생성 - PRD 예시 코드에 맞춘 버전
        Args:
            image_paths: 이미지 파일 경로 리스트
            batch_size: 사용하지 않음 (이전 버전과의 호환성을 위해 유지)
            progress_callback: 진행 상태를 보고하는 콜백 함수
            save_interval: 중간 저장할 이미지 간격 (기본값 50, 0이면 중간 저장 안함)
            temp_output_dir: 중간 결과 저장 디렉토리 (None이면 임시 디렉토리 사용)
            max_image_size: 이미지 최대 크기 (px, None이면 리사이즈 안함)
            low_memory_mode: 저메모리 모드 활성화 (기본값 True)
        Returns:
            이미지 경로와 임베딩을 포함한 딕셔너리
        """
        # 모델 로드 확인
        if progress_callback:
            progress_callback(0, len(image_paths), "모델 로딩 중...")
            
        if self.model is None or self.processor is None:
            if not self.load_model():
                if progress_callback:
                    progress_callback(0, len(image_paths), "모델 로딩 실패")
                return {}
        
        # 결과를 저장할 리스트 (딕셔너리 대신 메모리 효율적인 리스트 사용)
        file_names = []
        page_nums = []
        doc_ids = []
        embeddings = []
        
        # 중간 저장 디렉토리 설정
        temp_dir = None
        if save_interval > 0:
            import tempfile
            import shutil
            
            if temp_output_dir:
                temp_dir = Path(temp_output_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
            else:
                temp_dir = Path(tempfile.mkdtemp(prefix="embeddings_temp_"))
            
            print(f"중간 결과 저장 디렉토리: {temp_dir}")
        
        # 이미지 처리 - PRD 예시 코드에 맞게 구현
        from app.utils.pdf import extract_page_info_from_filename
        
        total_images = len(image_paths)
        processed_count = 0
        skip_count = 0
        error_count = 0
        
        try:
            # 이미 처리된 파일 목록 (중복 방지)
            processed_files = set()
            
            # 디바이스 설정
            device = self.device
            print(f"사용 중인 디바이스: {device}")
            
            # 한 번에 한 이미지씩 처리 (PRD 예시 코드와 유사하게 구현)
            for img_idx, img_path in enumerate(image_paths):
                if progress_callback:
                    progress_callback(img_idx, total_images, f"이미지 처리 중: {img_idx+1}/{total_images}")
                
                img_path = Path(img_path)
                
                # 이미 처리된 파일 스킵 
                if str(img_path) in processed_files:
                    skip_count += 1
                    continue
                    
                file_name = img_path.stem
                page_info = extract_page_info_from_filename(file_name)
                
                if page_info:
                    doc_id, page_num = page_info
                else:
                    # 페이지 정보를 추출할 수 없는 경우
                    doc_id = img_path.stem
                    page_num = 1
                
                try:
                    # 이미지 로드 및 전처리
                    img = Image.open(img_path).convert("RGB")
                    
                    # 이미지가 너무 크면 리사이즈
                    img = self.resize_image_if_needed(img, max_size=max_image_size)
                    
                    # 단일 이미지만 처리 - PRD 예시 코드와 일치하게 구현
                    processed_img = self.processor.process_images([img])
                    processed_img = {k: v.to(device) for k, v in processed_img.items()}
                    
                    # 임베딩 생성
                    with torch.no_grad():
                        embedding = self.model(**processed_img)
                    
                    # PRD의 예시 코드와 일치하도록 임베딩 수정
                    # CPU로 이동하여 저장 (메모리 절약)
                    # PRD 코드: ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
                    # 중요: unbind를 통해 차원을 변환해야 쿼리 임베딩과 호환됨
                    unbinded_embeddings = list(torch.unbind(embedding.to("cpu").detach()))
                    embeddings.extend(unbinded_embeddings)
                    
                    # 각 임베딩에 대한 메타데이터 복제 추가
                    for _ in range(len(unbinded_embeddings)):
                        file_names.append(str(img_path))
                        page_nums.append(page_num)
                        doc_ids.append(doc_id)
                    
                    # 처리된 파일 추적
                    processed_files.add(str(img_path))
                    
                    # 메모리 정리
                    del img, processed_img, embedding
                    
                    # 저메모리 모드에서는 매 이미지마다 메모리 정리
                    if low_memory_mode:
                        self.clear_memory()
                    
                    processed_count += 1
                    
                    # 중간 저장
                    if save_interval > 0 and temp_dir and processed_count % save_interval == 0:
                        temp_file = temp_dir / f"embeddings_temp_{processed_count}.pt"
                        if progress_callback:
                            progress_callback(img_idx+1, total_images, f"중간 결과 저장 중... ({processed_count}개)")
                        
                        # 중간 저장 시 딕셔너리로 변환
                        temp_results = {
                            "file_names": file_names,
                            "page_nums": page_nums,
                            "embeddings": embeddings,
                            "doc_ids": doc_ids
                        }
                        
                        torch.save(temp_results, temp_file)
                        print(f"중간 결과 저장 완료: {temp_file} ({processed_count}개)")
                        
                        # 임베딩 형태 검증
                        print(f"저장된 임베딩 첫 항목 shape: {embeddings[0].shape}")
                        
                        # 중간 저장 후 메모리 정리
                        self.clear_memory()
                    
                    if progress_callback:
                        progress_callback(img_idx+1, total_images, f"이미지 처리 완료: {processed_count}/{total_images}")
                    
                except Exception as e:
                    print(f"이미지 '{img_path}' 처리 중 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    
            # 최종 임베딩 형태 출력
            if len(embeddings) > 0:
                print(f"생성된 임베딩 shape: {embeddings[0].shape}")
                    
            # 최종 결과 저장 (중간 저장 사용했을 경우)
            if save_interval > 0 and temp_dir and processed_count > 0:
                final_file = temp_dir / "embeddings_final.pt"
                
                # 최종 저장 시 딕셔너리로 변환
                final_results = {
                    "file_names": file_names,
                    "page_nums": page_nums,
                    "embeddings": embeddings,
                    "doc_ids": doc_ids
                }
                
                torch.save(final_results, final_file)
                print(f"최종 결과 저장 완료: {final_file} ({processed_count}개)")
            
            if progress_callback:
                status_msg = f"총 {processed_count}개 이미지 임베딩 생성 완료"
                if skip_count > 0:
                    status_msg += f", {skip_count}개 스킵"
                if error_count > 0:
                    status_msg += f", {error_count}개 오류"
                
                progress_callback(total_images, total_images, status_msg)
            
            # 최종 결과를 딕셔너리로 반환
            return {
                "file_names": file_names,
                "page_nums": page_nums,
                "embeddings": embeddings,
                "doc_ids": doc_ids
            }
            
        except Exception as e:
            print(f"전체 처리 과정 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
            # 중간 저장한 결과가 있으면 마지막 중간 저장 결과 반환
            if save_interval > 0 and temp_dir:
                latest_temp_files = sorted(temp_dir.glob("embeddings_temp_*.pt"))
                if latest_temp_files:
                    latest_file = latest_temp_files[-1]
                    print(f"최신 중간 저장 결과 로드 중: {latest_file}")
                    results = torch.load(latest_file)
                    print(f"중간 저장 결과 로드 완료: {len(results['embeddings'])}개 임베딩")
                    return results
            
            # 중간 저장 결과가 없으면 지금까지 처리한 결과 반환
            if processed_count > 0:
                return {
                    "file_names": file_names,
                    "page_nums": page_nums,
                    "embeddings": embeddings,
                    "doc_ids": doc_ids
                }
            
            # 아무것도 처리하지 못한 경우 빈 결과 반환
            return {
                "file_names": [],
                "page_nums": [],
                "embeddings": [],
                "doc_ids": []
            }
    
    def save_embeddings(self, 
                        embeddings_data: Dict[str, Any], 
                        output_path: Union[str, Path]) -> bool:
        """
        임베딩 저장
        
        Args:
            embeddings_data: 임베딩 데이터 딕셔너리
            output_path: 저장할 디렉토리 경로
            
        Returns:
            저장 성공 여부
        """
        try:
            # 경로 객체 변환
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 임베딩 파일 경로
            embedding_file = output_path / EMBEDDING_FILE
            
            # 임베딩 데이터 검증
            if not embeddings_data or "embeddings" not in embeddings_data or not embeddings_data["embeddings"]:
                print(f"저장할 임베딩 데이터가 비어있습니다")
                return False
                
            embedding_count = len(embeddings_data["embeddings"])
            
            # 임베딩 저장 전 정보 출력
            print(f"임베딩 저장 중: {embedding_count}개 임베딩 → {embedding_file}")
            if embedding_count > 0:
                first_embedding = embeddings_data["embeddings"][0]
                print(f"첫 번째 임베딩 shape: {first_embedding.shape}")
            
            # 임베딩 저장
            torch.save(embeddings_data, embedding_file)
            
            # 저장 후 검증
            if embedding_file.exists():
                file_size = embedding_file.stat().st_size / (1024 * 1024)  # MB 단위
                print(f"임베딩 파일 저장 완료: {embedding_file} ({file_size:.2f} MB)")
                return True
            else:
                print(f"임베딩 파일이 생성되지 않았습니다: {embedding_file}")
                return False
        except Exception as e:
            print(f"임베딩 저장 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_partial_embeddings(self, embeddings_data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """
        부분 임베딩 저장
        
        Args:
            embeddings_data: 임베딩 데이터 딕셔너리
            file_path: 저장할 파일 경로
            
        Returns:
            저장 성공 여부
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(embeddings_data, file_path)
            return True
        except Exception as e:
            print(f"부분 임베딩 저장 중 오류 발생: {e}")
            return False
    
    def merge_embeddings(self, embedding_files: List[Union[str, Path]]) -> Optional[Dict[str, Any]]:
        """
        여러 임베딩 파일 병합
        
        Args:
            embedding_files: 임베딩 파일 경로 리스트
            
        Returns:
            병합된 임베딩 데이터 딕셔너리 또는 오류 시 None
        """
        try:
            merged_results = {
                "file_names": [],
                "page_nums": [],
                "embeddings": [],
                "doc_ids": []
            }
            
            for file_path in embedding_files:
                data = torch.load(file_path)
                if not data or not all(k in data for k in merged_results.keys()):
                    print(f"유효하지 않은 임베딩 파일: {file_path}")
                    continue
                
                merged_results["file_names"].extend(data["file_names"])
                merged_results["page_nums"].extend(data["page_nums"])
                merged_results["embeddings"].extend(data["embeddings"])
                merged_results["doc_ids"].extend(data["doc_ids"])
            
            return merged_results
        except Exception as e:
            print(f"임베딩 병합 중 오류 발생: {e}")
            return None
    
    def load_embeddings(self, embedding_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        저장된 임베딩 로드
        
        Args:
            embedding_dir: 임베딩 디렉토리 경로
            
        Returns:
            임베딩 데이터 딕셔너리 또는 오류 시 None
        """
        try:
            # 경로 객체 변환
            embedding_dir = Path(embedding_dir)
            embedding_file = embedding_dir / EMBEDDING_FILE
            
            if not embedding_file.exists():
                print(f"임베딩 파일이 존재하지 않습니다: {embedding_file}")
                return None
            
            # 임베딩 로드
            embeddings_data = torch.load(embedding_file)
            
            # 로드된 임베딩 정보 출력 (디버깅용)
            if embeddings_data and "embeddings" in embeddings_data and len(embeddings_data["embeddings"]) > 0:
                total_embeddings = len(embeddings_data["embeddings"])
                print(f"임베딩 로드 완료: 총 {total_embeddings}개")
                
                # 첫 번째 임베딩의 shape 출력
                first_emb = embeddings_data["embeddings"][0]
                print(f"첫 번째 임베딩 shape: {first_emb.shape}")
                
                # 임의의 몇 개 임베딩 shape도 확인 (모두 동일한지 검증)
                if total_embeddings > 1:
                    check_indices = [
                        0, 
                        total_embeddings // 2,  # 중간
                        total_embeddings - 1    # 마지막
                    ]
                    shapes_consistent = True
                    first_shape = first_emb.shape
                    
                    for idx in check_indices[1:]:  # 첫 번째는 이미 확인했으므로 건너뜀
                        curr_shape = embeddings_data["embeddings"][idx].shape
                        if curr_shape != first_shape:
                            shapes_consistent = False
                            print(f"경고: 임베딩[{idx}]의 shape({curr_shape})가 첫 번째 임베딩({first_shape})과 다릅니다.")
                    
                    if shapes_consistent:
                        print(f"모든 임베딩이 동일한 shape({first_shape})를 가집니다.")
            
            return embeddings_data
        except Exception as e:
            print(f"임베딩 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_embedding_stats(self, embedding_dir: Union[str, Path]) -> Dict[str, int]:
        """
        임베딩 통계 정보 반환
        
        Args:
            embedding_dir: 임베딩 디렉토리 경로
            
        Returns:
            통계 정보 딕셔너리 (임베딩 수 등)
        """
        stats = {
            "total_embeddings": 0,
            "unique_docs": 0
        }
        
        embeddings_data = self.load_embeddings(embedding_dir)
        if embeddings_data and "file_names" in embeddings_data:
            stats["total_embeddings"] = len(embeddings_data["file_names"])
            
            if "doc_ids" in embeddings_data:
                stats["unique_docs"] = len(set(embeddings_data["doc_ids"]))
        
        return stats 