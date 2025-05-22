"""
Knowledge 관리 UI 관련 모듈
"""
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from app.models.knowledge import KnowledgeModel
from app.utils.constants import KNOWLEDGE_DIR

def render_knowledge_view(knowledge_id: str = None):
    """
    Knowledge 관리 UI 렌더링
    
    Args:
        knowledge_id: 선택된 Knowledge ID (없으면 None)
    """
    if not knowledge_id:
        st.warning("먼저 사이드바에서 Knowledge를 선택하거나 생성해주세요.")
        return
    
    st.header("Knowledge 정보")
    
    # Knowledge 정보 가져오기
    knowledge_info = KnowledgeModel.get_knowledge_info(knowledge_id)
    
    if not knowledge_info:
        st.error(f"Knowledge 정보를 불러올 수 없습니다: {knowledge_id}")
        return
    
    # 메트릭 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pdf_files = KnowledgeModel.list_pdf_files(knowledge_id)
        pdf_count = len(pdf_files) if pdf_files else 0
        st.metric("PDF 파일 수", pdf_count)
    
    with col2:
        image_files = KnowledgeModel.list_images(knowledge_id)
        image_count = len(image_files) if image_files else 0
        st.metric("이미지 파일 수", image_count)
    
    with col3:
        # 임베딩 통계
        from app.models.embedding import EmbeddingManager
        embedding_manager = EmbeddingManager()
        embedding_stats = embedding_manager.get_embedding_stats(
            KnowledgeModel.get_embedding_path(knowledge_id)
        )
        embedding_count = embedding_stats.get("total_embeddings", 0)
        st.metric("임베딩 수", embedding_count)
    
    # Knowledge 상세 정보
    with st.expander("Knowledge 상세 정보", expanded=True):
        st.markdown(f"**이름**: {knowledge_info.name}")
        st.markdown(f"**설명**: {knowledge_info.description or 'N/A'}")
        st.markdown(f"**생성일**: {knowledge_info.created_at}")
        st.markdown(f"**마지막 수정일**: {knowledge_info.updated_at}")
        st.markdown(f"**ID**: {knowledge_id}")
        st.markdown(f"**경로**: {KNOWLEDGE_DIR / knowledge_id}")
    
    # PDF 파일 요약
    if pdf_count > 0:
        with st.expander("PDF 파일 목록", expanded=False):
            # PDF 파일 정보 표시
            pdf_data = []
            for pdf_file in pdf_files:
                pdf_data.append({
                    "파일명": pdf_file.name,
                    "크기(KB)": round(pdf_file.stat().st_size / 1024, 2),
                    "수정일": pd.Timestamp(pdf_file.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')
                })
            
            # 데이터프레임으로 변환하여 표시
            df_pdfs = pd.DataFrame(pdf_data)
            st.dataframe(df_pdfs, use_container_width=True)

def show_knowledge_management():
    """Knowledge 관리 UI 컴포넌트"""
    st.header("Knowledge 관리")
    
    # Knowledge 생성 UI
    st.subheader("새 Knowledge 생성")
    with st.form("create_knowledge"):
        name = st.text_input("Knowledge 이름")
        description = st.text_area("설명 (선택사항)")
        create_button = st.form_submit_button("생성")
        
        if create_button and name:
            # TODO: Knowledge 생성 로직 연결
            try:
                knowledge = KnowledgeModel.create_knowledge(name, description)
                st.success(f"Knowledge '{name}'가 생성되었습니다. (ID: {knowledge.id})")
            except Exception as e:
                st.error(f"Knowledge 생성 중 오류 발생: {e}")
        elif create_button:
            st.warning("Knowledge 이름을 입력하세요.")
    
    # Knowledge 선택 UI
    st.subheader("Knowledge 선택")
    
    # 현재는 더미 데이터
    # TODO: 실제 Knowledge 목록 조회 로직 연결
    knowledge_list = [
        {"id": "k_1", "name": "예제 Knowledge 1", "pdf_count": 3, "image_count": 15, "embedding_count": 15},
        {"id": "k_2", "name": "예제 Knowledge 2", "pdf_count": 2, "image_count": 10, "embedding_count": 10},
    ]
    
    if not knowledge_list:
        st.info("생성된 Knowledge가 없습니다. 위에서 Knowledge를 생성해주세요.")
    else:
        knowledge_names = [f"{k['name']} (PDF: {k['pdf_count']}, Images: {k['image_count']})" for k in knowledge_list]
        selected_idx = st.selectbox("Knowledge를 선택하세요", range(len(knowledge_names)), format_func=lambda i: knowledge_names[i])
        
        if selected_idx is not None:
            selected_knowledge = knowledge_list[selected_idx]
            st.session_state["current_knowledge_id"] = selected_knowledge["id"]
            st.session_state["current_knowledge_name"] = selected_knowledge["name"]
            
            # Knowledge 상세 정보
            st.write(f"**ID:** {selected_knowledge['id']}")
            st.write(f"**PDF 파일 수:** {selected_knowledge['pdf_count']}")
            st.write(f"**이미지 수:** {selected_knowledge['image_count']}")
            st.write(f"**임베딩 수:** {selected_knowledge['embedding_count']}")

def get_current_knowledge():
    """현재 선택된 Knowledge ID 반환"""
    return st.session_state.get("current_knowledge_id", None)

def check_knowledge_selected():
    """Knowledge가 선택되었는지 확인"""
    if "current_knowledge_id" not in st.session_state:
        st.warning("작업할 Knowledge를 선택하세요.")
        return False
    return True 