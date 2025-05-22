"""
검색 UI 관련 모듈
"""
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
import pandas as pd
import time
import tempfile
import os
import json
import unicodedata

from app.models.embedding import EmbeddingManager
from app.utils.query import process_query, find_similar_images, normalize_scores, parse_query_file
from app.utils.pdf import extract_page_info_from_filename


def render_search_ui(knowledge_id: str):
    """
    검색 UI 렌더링
    
    Args:
        knowledge_id: Knowledge ID
    """
    if not knowledge_id:
        st.warning("먼저 사이드바에서 Knowledge를 선택하거나 생성해주세요.")
        return
    
    st.header("이미지 검색")
    
    # 탭 생성: 단일 검색과 배치 검색
    tabs = st.tabs(["직접 검색", "파일 일괄 검색"])
    
    # 탭 1: 직접 검색 (기존 기능)
    with tabs[0]:
        render_single_search(knowledge_id)
    
    # 탭 2: 파일 일괄 검색 (새 기능)
    with tabs[1]:
        render_batch_search(knowledge_id)


def render_single_search(knowledge_id: str):
    """
    단일 검색 UI 렌더링
    
    Args:
        knowledge_id: Knowledge ID
    """
    # 로딩 스피너를 위한 컨테이너
    loading_container = st.container()
    
    # 검색 정보 안내
    st.info("이 검색은 ColQwen 모델의 processor를 사용하여 정확한 유사도를 계산합니다. 검색과 이미지 모두 동일한 임베딩 차원이 보장됩니다.")
    
    # 검색 설정 컬럼
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 쿼리 입력
        query = st.text_input("질문 또는 검색어를 입력하세요", 
                            help="찾고자 하는 정보를 질문 형태로 입력하세요.")
    
    with col2:
        # 검색 설정
        top_k = st.slider("표시할 결과 수", min_value=1, max_value=20, value=5,
                        help="검색 결과로 표시할 이미지 개수")
    
    # 검색 버튼
    search_button = st.button("검색")
    
    # 임베딩 상태 확인
    from app.models.knowledge import KnowledgeModel
    embedding_dir = KnowledgeModel.get_embedding_path(knowledge_id)
    embeddings_data = None
    
    # 검색 수행
    if search_button and query:
        with st.spinner("검색 중..."):
            start_time = time.time()
            
            # EmbeddingManager 초기화
            try:
                embedding_manager = EmbeddingManager()
                
                # 임베딩 로드
                with loading_container:
                    with st.status("검색 진행 중...", expanded=False) as status:
                        status.update(label="임베딩 데이터 로드 중...")
                        embeddings_data = embedding_manager.load_embeddings(embedding_dir)
                        
                        if embeddings_data is None or "embeddings" not in embeddings_data or not embeddings_data["embeddings"]:
                            st.warning("이 Knowledge에 저장된 임베딩이 없습니다. 먼저 이미지 탭에서 임베딩을 생성해주세요.")
                            status.update(label="검색 실패", state="error")
                            return
                        
                        total_embeddings = len(embeddings_data["embeddings"])
                        st.info(f"총 {total_embeddings}개의 이미지 임베딩이 로드되었습니다.")
                        
                        # 문서 ID 필터링
                        doc_filter = st.text_input("문서 ID 필터", 
                                 help="특정 문서 ID만 검색 (빈칸이면 전체 검색)")
                        if doc_filter:
                            filtered_indices = [i for i, doc_id in enumerate(embeddings_data["doc_ids"]) if doc_filter in doc_id]
                            
                            if not filtered_indices:
                                st.warning(f"필터링 조건 '{doc_filter}'에 해당하는 문서가 없습니다.")
                                status.update(label="검색 실패", state="error")
                                return
                            
                            # 필터링된 임베딩 데이터 생성
                            filtered_data = {
                                "file_names": [embeddings_data["file_names"][i] for i in filtered_indices],
                                "page_nums": [embeddings_data["page_nums"][i] for i in filtered_indices],
                                "embeddings": [embeddings_data["embeddings"][i] for i in filtered_indices],
                                "doc_ids": [embeddings_data["doc_ids"][i] for i in filtered_indices]
                            }
                            
                            embeddings_data = filtered_data
                            st.info(f"필터링 결과: {len(filtered_indices)}개의 이미지가 대상입니다.")
                        
                        # 모델 로드 확인
                        status.update(label="모델 로드 중...")
                        if embedding_manager.model is None or embedding_manager.processor is None:
                            success = embedding_manager.load_model()
                            if not success:
                                st.error("검색 모델을 로드할 수 없습니다.")
                                status.update(label="검색 실패", state="error")
                                return
                        
                        # 쿼리 임베딩 생성
                        status.update(label="쿼리 임베딩 생성 중...")
                        query_embedding = process_query(
                            query_text=query,
                            model=embedding_manager.model,
                            processor=embedding_manager.processor,
                            device=embedding_manager.device
                        )
                        
                        if query_embedding is None:
                            st.error("쿼리 임베딩을 생성할 수 없습니다.")
                            status.update(label="검색 실패", state="error")
                            return
                        
                        # 유사 이미지 검색
                        status.update(label="유사 이미지 검색 중...")
                        
                        results = find_similar_images(
                            query_embedding=query_embedding,
                            image_embeddings_data=embeddings_data,
                            top_k=top_k,
                            processor=embedding_manager.processor,
                            similarity_method="processor"  # processor만 사용
                        )
                        
                        # 점수 정규화
                        if results:
                            results = normalize_scores(results)
                        
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                        status.update(label=f"검색 완료 ({elapsed_time:.2f}초)", state="complete")
                
                # 검색 결과 표시
                render_search_results(query, results, elapsed_time)
                
            except Exception as e:
                st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return


def render_batch_search(knowledge_id: str):
    """
    CSV/JSON 파일을 통한 일괄 검색 UI 렌더링
    
    Args:
        knowledge_id: Knowledge ID
    """
    st.subheader("CSV/JSON 파일 일괄 검색")
    st.info("CSV 또는 JSON 파일을 업로드하여 여러 질문을 한 번에 검색할 수 있습니다.")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("CSV 또는 JSON 파일 업로드", type=["csv", "json"])
    
    if uploaded_file is not None:
        # 임시 파일로 저장
        temp_dir = Path(tempfile.mkdtemp())
        file_path = temp_dir / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # 파일 형식에 따라 미리보기 표시
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
                st.write("파일 미리보기:")
                st.dataframe(df.head())
                
                # 컬럼 목록 가져오기
                columns = df.columns.tolist()
                
            elif uploaded_file.name.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                st.write("파일 미리보기:")
                st.json(data if isinstance(data, dict) else data[:3])
                
                # JSON 구조에 따라 키 목록 가져오기
                if isinstance(data, list) and len(data) > 0:
                    columns = list(data[0].keys())
                elif isinstance(data, dict):
                    columns = list(data.keys())
                else:
                    columns = []
            
            # 컬럼 매핑 설정
            st.subheader("컬럼 매핑")
            st.info("검색에 사용할 컬럼이나 키를 지정해주세요.")
            
            # 필수: 질문 컬럼
            question_col = st.selectbox(
                "질문 컬럼 선택 (필수)", 
                options=[col for col in columns if "question" in col.lower()] + columns,
                help="검색 질문이 포함된 컬럼이나 키를 선택하세요."
            )
            
            # 선택: 대상 파일
            target_file_col = st.selectbox(
                "대상 파일 컬럼 선택 (선택 사항)", 
                options=["없음"] + [col for col in columns if "file" in col.lower()] + columns,
                help="특정 파일에서만 검색할 경우 해당 컬럼을 선택하세요."
            )
            target_file_col = None if target_file_col == "없음" else target_file_col
            
            # 선택: 대상 페이지
            target_page_col = st.selectbox(
                "대상 페이지 컬럼 선택 (선택 사항)", 
                options=["없음"] + [col for col in columns if "page" in col.lower()] + columns,
                help="특정 페이지에서만 검색할 경우 해당 컬럼을 선택하세요."
            )
            target_page_col = None if target_page_col == "없음" else target_page_col
            
            # 검색 설정
            st.subheader("검색 설정")
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_k = st.slider("질문당 표시할 결과 수", min_value=1, max_value=20, value=3,
                                help="각 질문당 상위 몇 개의 결과를 표시할지 선택")
            
            with col2:
                normalize = st.checkbox("점수 정규화", value=True, key="normalize_batch_search",
                                       help="검색 결과 점수를 0~1 범위로 정규화합니다.")
            
            # 검색 실행 버튼
            run_batch_search = st.button("일괄 검색 실행")
            
            if run_batch_search:
                # 파일 로딩
                st.subheader("검색 결과")
                
                # 임베딩 로드
                embedding_manager = EmbeddingManager()
                from app.models.knowledge import KnowledgeModel
                embedding_dir = KnowledgeModel.get_embedding_path(knowledge_id)
                
                with st.spinner("임베딩 로드 중..."):
                    embeddings_data = embedding_manager.load_embeddings(embedding_dir)
                    
                    if embeddings_data is None or "embeddings" not in embeddings_data or not embeddings_data["embeddings"]:
                        st.warning("이 Knowledge에 저장된 임베딩이 없습니다. 먼저 이미지 탭에서 임베딩을 생성해주세요.")
                        return
                    
                    total_embeddings = len(embeddings_data["embeddings"])
                    st.info(f"총 {total_embeddings}개의 이미지 임베딩이 로드되었습니다.")
                
                # 모델 로드
                with st.spinner("모델 로드 중..."):
                    if embedding_manager.model is None or embedding_manager.processor is None:
                        success = embedding_manager.load_model()
                        if not success:
                            st.error("검색 모델을 로드할 수 없습니다.")
                            return
                
                # 질문 추출
                questions = []
                target_files = []
                target_pages = []
                
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        
                        # 질문 컬럼 유효성 검사
                        if question_col not in df.columns:
                            st.error(f"선택한 질문 컬럼 '{question_col}'이 CSV 파일에 존재하지 않습니다.")
                            return
                        
                        questions = df[question_col].tolist()
                        
                        # 대상 파일 컬럼이 있으면 추출
                        if target_file_col and target_file_col in df.columns:
                            target_files = df[target_file_col].tolist()
                        else:
                            target_files = [None] * len(questions)
                        
                        # 대상 페이지 컬럼이 있으면 추출
                        if target_page_col and target_page_col in df.columns:
                            target_pages = df[target_page_col].tolist()
                        else:
                            target_pages = [None] * len(questions)
                    
                    elif uploaded_file.name.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        if isinstance(data, list):
                            # 리스트 형태의 JSON
                            for item in data:
                                if question_col in item:
                                    questions.append(item[question_col])
                                    
                                    # 대상 파일
                                    if target_file_col and target_file_col in item:
                                        target_files.append(item[target_file_col])
                                    else:
                                        target_files.append(None)
                                    
                                    # 대상 페이지
                                    if target_page_col and target_page_col in item:
                                        target_pages.append(item[target_page_col])
                                    else:
                                        target_pages.append(None)
                        
                        elif isinstance(data, dict):
                            # 딕셔너리 형태의 JSON
                            if question_col in data and isinstance(data[question_col], list):
                                questions = data[question_col]
                                
                                # 대상 파일
                                if target_file_col and target_file_col in data and isinstance(data[target_file_col], list):
                                    target_files = data[target_file_col]
                                else:
                                    target_files = [None] * len(questions)
                                
                                # 대상 페이지
                                if target_page_col and target_page_col in data and isinstance(data[target_page_col], list):
                                    target_pages = data[target_page_col]
                                else:
                                    target_pages = [None] * len(questions)
                            else:
                                st.error(f"선택한 질문 컬럼 '{question_col}'이 JSON 파일에 리스트 형태로 존재하지 않습니다.")
                                return
                
                except Exception as e:
                    st.error(f"파일 처리 중 오류 발생: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    return
                
                if not questions:
                    st.error("파일에서 질문을 추출할 수 없습니다.")
                    return
                
                st.info(f"총 {len(questions)}개의 질문을 처리합니다.")
                
                # 검색 프로그레스 바
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 결과 저장 컨테이너
                all_results = []
                
                # 각 질문에 대해 검색 수행
                with st.spinner("일괄 검색 진행 중..."):
                    for i, question in enumerate(questions):
                        status_text.text(f"검색 중: {i+1}/{len(questions)} - {question}")
                        progress_bar.progress((i) / len(questions))
                        
                        # 쿼리 임베딩 생성
                        query_embedding = process_query(
                            query_text=question,
                            model=embedding_manager.model,
                            processor=embedding_manager.processor,
                            device=embedding_manager.device
                        )
                        
                        if query_embedding is None:
                            st.warning(f"질문 '{question}'의 임베딩을 생성할 수 없습니다. 건너뜁니다.")
                            continue
                        
                        # 검색 실행
                        results = find_similar_images(
                            query_embedding=query_embedding,
                            image_embeddings_data=embeddings_data,
                            top_k=top_k,
                            processor=embedding_manager.processor,
                            similarity_method="processor"
                        )
                        
                        # 정규화
                        if normalize and results:
                            results = normalize_scores(results)
                        
                        # 결과 저장
                        if results:
                            result_entry = {
                                "question": question,
                                "target_file": target_files[i],
                                "target_page": target_pages[i],
                                "results": results
                            }
                            all_results.append(result_entry)
                    
                    # 완료 표시
                    progress_bar.progress(1.0)
                    status_text.text(f"검색 완료: {len(all_results)}/{len(questions)} 질문 처리됨")
                
                # 결과 표시
                if all_results:
                    # 1. 정답 판정 및 통계
                    correct_count = 0
                    for result in all_results:
                        # 각 결과의 실제 파일명/페이지 추출 (expander에서는 더이상 사용하지 않음)
                        for r in result["results"]:
                            from app.utils.pdf import extract_page_info_from_filename
                            page_info = extract_page_info_from_filename(Path(r["file_name"]).name)
                            if page_info:
                                r["real_file_name"] = page_info[0]
                                r["real_page_num"] = page_info[1]
                            else:
                                r["real_file_name"] = Path(r["file_name"]).name
                                r["real_page_num"] = r.get("page_num", "?")
                        # 정답 판정
                        is_correct = False
                        def norm(s):
                            if s is None:
                                return None
                            return unicodedata.normalize('NFC', str(s))
                        def extract_base_pdf_name(image_file_name):
                            # 예: "문서명_page_1.png" → "문서명"
                            stem = Path(image_file_name).stem
                            if "_page_" in stem:
                                base = stem.split("_page_")[0]
                                return base
                            return stem
                        for r in result["results"]:
                            # 파일명 비교: 확장자/페이지 제외, 유니코드 normalize
                            file_match = (
                                result["target_file"] is None or
                                norm(extract_base_pdf_name(r["real_file_name"])) == norm(Path(str(result["target_file"])).stem)
                            )
                            # 페이지 비교: int로 캐스팅
                            try:
                                page_match = (
                                    result["target_page"] is None or
                                    int(r["real_page_num"]) == int(result["target_page"])
                                )
                            except Exception:
                                page_match = False
                            if file_match and page_match:
                                is_correct = True
                                break
                        result["is_correct"] = is_correct
                        if is_correct:
                            correct_count += 1
                    accuracy = correct_count / len(all_results) * 100 if all_results else 0.0

                    # summary 테이블 데이터 가공
                    summary_data = []
                    for idx, result in enumerate(all_results):
                        # top-k 검색결과 파일명/페이지 리스트 생성
                        file_names = [r["real_file_name"] for r in result["results"]]
                        page_nums = [r["real_page_num"] for r in result["results"]]
                        summary_data.append({
                            "질문": result["question"],
                            "정답": "정답" if result["is_correct"] else "오답",
                            "검색결과 파일명": file_names,
                            "검색결과 페이지": page_nums,
                            "target_file": result["target_file"],
                            "target_page": result["target_page"]
                        })

                    st.markdown(f"### 정답률: **{accuracy:.1f}%** ({correct_count}/{len(all_results)})")
                    st.dataframe(summary_data, hide_index=True)

                    # 상세 결과(expander) 테이블에서 '실제파일명', '실제페이지' 컬럼 제거
                    for idx, result in enumerate(all_results):
                        with st.expander(f"[{idx+1}] 질문: {result['question']} ({'정답' if result['is_correct'] else '오답'})"):
                            result_table = []
                            for r in result["results"]:
                                result_table.append({
                                    "파일명": r["real_file_name"],
                                    "페이지": r["real_page_num"],
                                    "유사도": r["score"] if not (r["score"] is None or str(r["score"]) == "nan") else 0.0
                                })
                            st.dataframe(result_table, hide_index=True)
                            # 이미지 등 기타 표시 로직은 기존대로 유지
                
                else:
                    st.warning("모든 질문에 대해 검색 결과가 없습니다.")
                
                # 임시 디렉토리 정리
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        except Exception as e:
            st.error(f"파일 처리 중 오류 발생: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            
            # 임시 디렉토리 정리
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass


def render_search_results(query: str, results: List[Dict[str, Any]], elapsed_time: float = None):
    """
    검색 결과 표시
    
    Args:
        query: 검색 쿼리
        results: 검색 결과 리스트
        elapsed_time: 검색 소요 시간
    """
    # 결과 요약
    if not results:
        st.warning("검색 결과가 없습니다.")
        return
    
    if elapsed_time:
        st.success(f"검색 결과: {len(results)}개의 관련 이미지를 찾았습니다. (소요 시간: {elapsed_time:.2f}초)")
    else:
        st.success(f"검색 결과: {len(results)}개의 관련 이미지를 찾았습니다.")
    
    # 데이터 테이블로 요약 보기
    df_results = pd.DataFrame([
        {
            "파일명": Path(r["file_name"]).name,
            "문서 ID": r["doc_id"],
            "페이지": r["page_num"],
            "유사도": f"{r['score']:.4f}"
        } for r in results
    ])
    
    with st.expander("결과 요약 테이블", expanded=False):
        st.dataframe(df_results, use_container_width=True)
    
    # 결과 카드 형태로 표시
    for i, result in enumerate(results):
        with st.container():
            cols = st.columns([1, 2])
            
            # 왼쪽 컬럼: 이미지
            with cols[0]:
                try:
                    image_path = result["file_name"]
                    st.image(image_path, caption=f"Score: {result['score']:.4f}", use_container_width=True)
                except Exception as e:
                    st.error(f"이미지 로드 중 오류: {e}")
            
            # 오른쪽 컬럼: 정보
            with cols[1]:
                st.markdown(f"**파일명**: {Path(result['file_name']).name}")
                st.markdown(f"**문서 ID**: {result['doc_id']}")
                st.markdown(f"**페이지**: {result['page_num']}")
                st.markdown(f"**유사도 점수**: {result['score']:.4f}")
                
                # 원본 이미지 표시 컨트롤
                if st.button(f"원본 크기로 보기 #{i+1}", key=f"view_orig_{i}"):
                    with st.expander(f"원본 이미지 #{i+1}", expanded=True):
                        st.image(result["file_name"])
            
            st.divider()
    
    # 검색 관련 정보
    with st.expander("검색 정보", expanded=False):
        st.markdown(f"**검색어**: {query}")
        st.markdown(f"**검색 결과 수**: {len(results)}")
        if elapsed_time:
            st.markdown(f"**소요 시간**: {elapsed_time:.4f}초")
        st.markdown(f"**최고 유사도**: {results[0]['score']:.4f}")
        st.markdown(f"**최저 유사도**: {results[-1]['score']:.4f}") 