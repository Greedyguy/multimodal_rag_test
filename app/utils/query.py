"""
쿼리 처리 관련 유틸리티 함수
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import pandas as pd
import json
import csv
from tqdm import tqdm
import numpy as np

def process_query(query_text: str, model, processor, device="cuda:0") -> Optional[torch.Tensor]:
    """
    텍스트 쿼리 임베딩 생성 (PRD 예시 코드에 맞춤)
    
    Args:
        query_text: 쿼리 텍스트
        model: 사용할 모델
        processor: 사용할 프로세서
        device: 사용할 디바이스 (기본값: "cuda:0")
        
    Returns:
        쿼리 임베딩 텐서 또는 오류 시 None
    """
    try:
        # 쿼리 전처리
        processed_query = processor.process_queries([query_text])
        
        # 디바이스 확인 및 이동
        device = torch.device(device)
        processed_query = {k: v.to(device) for k, v in processed_query.items()}
        
        # 임베딩 생성
        with torch.no_grad():
            embedding = model(**processed_query)
        
        # 디버깅을 위한 정보 출력
        print(f"생성된 쿼리 임베딩 차원 정보: {embedding.shape}")
        
        # query 임베딩은 PRD 예시와 같이 원본 형태 그대로 반환
        # (get_image_embedding에서 unbind를 적용하므로 여기서는 적용하지 않음)
        return embedding
    except Exception as e:
        print(f"쿼리 임베딩 생성 중 오류 발생: {e}")
        # 상세 오류 정보 출력
        import traceback
        traceback.print_exc()
        return None

# 참고: compute_similarity 함수는 더 이상 사용되지 않습니다.
# 모든 유사도 계산은 processor.score_multi_vector()를 통해 수행됩니다.
# (지침: 모든 쿼리와 이미지는 동일한 ColQwen 모델에서 생성되어야 하며, processor를 통해 유사도를 계산해야 합니다.)

def find_similar_images(
    query_embedding: torch.Tensor,
    image_embeddings_data: Dict[str, Any],
    top_k: int = 5,
    processor=None,
    similarity_method: str = "processor"
) -> List[Dict[str, Any]]:
    """
    쿼리 임베딩과 가장 유사한 이미지 찾기 (PRD 예시 코드에 맞춤)
    
    Args:
        query_embedding: 쿼리 임베딩 텐서
        image_embeddings_data: 이미지 임베딩 데이터
        top_k: 반환할 상위 결과 수
        processor: 유사도 계산을 위한 프로세서 (필수)
        similarity_method: 더 이상 사용되지 않음 (processor만 지원)
        
    Returns:
        유사도 높은 순으로 상위 k개 이미지 정보 리스트
    """
    try:
        if not processor:
            raise ValueError("Processor가 필요합니다. processor=None이면 유사도 계산을 할 수 없습니다.")
            
        if not image_embeddings_data or "embeddings" not in image_embeddings_data or not image_embeddings_data["embeddings"]:
            print("이미지 임베딩 데이터가 비어있습니다.")
            return []
        
        # 임베딩 데이터 확인
        print(f"총 {len(image_embeddings_data['embeddings'])}개의 이미지 임베딩이 있습니다.")
        
        # 임베딩 차원 디버깅 정보 출력
        print(f"쿼리 임베딩 차원: {query_embedding.shape}")
        if len(image_embeddings_data["embeddings"]) > 0:
            print(f"첫 번째 이미지 임베딩 차원: {image_embeddings_data['embeddings'][0].shape}")
            
            # 문서 ID와 페이지 수도 출력 (검증용)
            print(f"임베딩 개수: {len(image_embeddings_data['embeddings'])}")
            print(f"파일명 개수: {len(image_embeddings_data['file_names'])}")
            print(f"페이지 번호 개수: {len(image_embeddings_data['page_nums'])}")
            print(f"문서 ID 개수: {len(image_embeddings_data['doc_ids'])}")
        
        # PRD 예시 코드에 맞게 구현: rag 함수 형태로 동작
        try:
            # PRD 코드 예시처럼 차원 조정없이 직접 processor.score_multi_vector 호출
            scores = processor.score_multi_vector(query_embedding, image_embeddings_data["embeddings"])
            scores = scores[0].sort(descending=True)
            
            # 상위 k개 결과 추출
            results = []
            for i in range(min(top_k, len(scores.values))):
                idx = scores.indices[i].item()  # 텐서에서 정수로 변환
                results.append({
                    "file_name": image_embeddings_data["file_names"][idx],
                    "page_num": image_embeddings_data["page_nums"][idx],
                    "score": float(scores.values[i]),
                    "doc_id": image_embeddings_data["doc_ids"][idx]
                })
            
            return results
            
        except Exception as e:
            print(f"Processor 유사도 계산 중 오류 발생: {e}")
            # 오류 세부 정보 확인을 위한 임베딩 차원 출력
            print(f"쿼리 임베딩 차원: {query_embedding.shape}")
            
            # 이미지 임베딩 차원 확인 (랜덤 샘플 5개 출력)
            sample_size = min(5, len(image_embeddings_data["embeddings"]))
            for i in range(sample_size):
                idx = np.random.randint(0, len(image_embeddings_data["embeddings"]))
                print(f"이미지 임베딩[{idx}] 차원: {image_embeddings_data['embeddings'][idx].shape}")
            
            print("임베딩 차원이 일치하지 않거나 형식이 다른 것 같습니다. 임베딩을 다시 생성하는 것을 권장합니다.")
            raise
    
    except Exception as e:
        print(f"유사 이미지 검색 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []

def normalize_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    검색 결과의 점수 정규화
    
    Args:
        results: 검색 결과 리스트
        
    Returns:
        점수가 정규화된 검색 결과 리스트
    """
    if not results:
        return []
    
    # 최소/최대 점수 찾기
    min_score = min(result["score"] for result in results)
    max_score = max(result["score"] for result in results)
    
    # 점수 차이가 없으면 그대로 반환
    if max_score == min_score:
        return results
    
    # 점수 정규화 (0~1 범위)
    normalized_results = []
    for result in results:
        normalized_result = result.copy()
        normalized_result["score"] = (result["score"] - min_score) / (max_score - min_score)
        normalized_results.append(normalized_result)
    
    return normalized_results

def parse_query_file(file_path: Union[str, Path]) -> Tuple[List[str], Optional[List[str]], Optional[List[int]]]:
    """
    쿼리 파일(CSV, JSON) 파싱
    
    Args:
        file_path: CSV 또는 JSON 파일 경로
        
    Returns:
        (질문 리스트, 타겟 파일 리스트(또는 None), 타겟 페이지 리스트(또는 None))
    """
    file_path = Path(file_path)
    questions = []
    target_files = None
    target_pages = None
    
    try:
        if file_path.suffix.lower() == '.csv':
            # CSV 파일 읽기
            df = pd.read_csv(file_path)
            
            # 컬럼 확인
            if 'question' not in df.columns:
                raise ValueError("CSV 파일에 'question' 컬럼이 없습니다.")
            
            questions = df['question'].tolist()
            
            # 선택적 컬럼 확인
            if 'target_file' in df.columns:
                target_files = df['target_file'].tolist()
            
            if 'target_page' in df.columns:
                target_pages = df['target_page'].tolist()
                
        elif file_path.suffix.lower() == '.json':
            # JSON 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 형식 확인 (리스트 또는 딕셔너리)
            if isinstance(data, list):
                # 리스트 형식
                questions = []
                target_files_tmp = []
                target_pages_tmp = []
                has_target_file = False
                has_target_page = False
                
                for item in data:
                    if 'question' not in item:
                        raise ValueError("JSON 항목에 'question' 필드가 없습니다.")
                    
                    questions.append(item['question'])
                    
                    if 'target_file' in item:
                        target_files_tmp.append(item['target_file'])
                        has_target_file = True
                    else:
                        target_files_tmp.append(None)
                    
                    if 'target_page' in item:
                        target_pages_tmp.append(item['target_page'])
                        has_target_page = True
                    else:
                        target_pages_tmp.append(None)
                
                if has_target_file:
                    target_files = target_files_tmp
                
                if has_target_page:
                    target_pages = target_pages_tmp
            
            elif isinstance(data, dict):
                # 딕셔너리 형식
                if 'questions' not in data:
                    raise ValueError("JSON 파일에 'questions' 필드가 없습니다.")
                
                questions = data['questions']
                
                if 'target_files' in data:
                    target_files = data['target_files']
                
                if 'target_pages' in data:
                    target_pages = data['target_pages']
            
            else:
                raise ValueError("지원되지 않는 JSON 형식입니다.")
        
        else:
            raise ValueError("지원되지 않는 파일 형식입니다. CSV 또는 JSON만 지원합니다.")
        
        return questions, target_files, target_pages
    
    except Exception as e:
        print(f"쿼리 파일 파싱 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return [], None, None 