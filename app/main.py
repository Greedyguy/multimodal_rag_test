"""
Multimodal RAG 시스템의 메인 Streamlit 애플리케이션
"""
import os
import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd
import time
import torch  # 임베딩 및 기기 정보용
import sys
from datetime import datetime

from app.models.knowledge import KnowledgeModel
from app.utils.constants import KNOWLEDGE_DIR, MODEL_NAME, DEFAULT_DPI

# 뷰 모듈 가져오기
from app.views.search_ui import render_search_ui, render_search_results
from app.views.knowledge_view import render_knowledge_view

def create_knowledge_sidebar():
    """
    사이드바에 Knowledge 관리 UI 구현
    """
    with st.sidebar:
        st.header("Knowledge 관리")
        
        # Knowledge 생성
        with st.expander("새 Knowledge 생성", expanded=False):
            with st.form("create_knowledge"):
                knowledge_name = st.text_input("Knowledge 이름")
                knowledge_desc = st.text_area("설명 (선택사항)")
                
                create_button = st.form_submit_button("생성")
                if create_button:
                    if not knowledge_name:
                        st.error("Knowledge 이름을 입력해주세요.")
                    else:
                        with st.spinner("Knowledge 생성 중..."):
                            knowledge_info = KnowledgeModel.create_knowledge(
                                name=knowledge_name,
                                description=knowledge_desc
                            )
                            st.success(f"Knowledge '{knowledge_name}'가 생성되었습니다.")
                            # 새로 생성된 Knowledge를 선택
                            st.session_state["selected_knowledge_id"] = knowledge_info.id
                            # 페이지 재로드를 통해 입력 필드 초기화
                            st.rerun()
        
        st.divider()
        
        # Knowledge 목록 및 선택
        st.subheader("Knowledge 선택")
        knowledge_list = KnowledgeModel.get_knowledge_infos()
        
        if not knowledge_list:
            st.info("생성된 Knowledge가 없습니다. 위에서 Knowledge를 생성해주세요.")
        else:
            # Knowledge 정보 수집 (파일 개수 등)
            knowledge_options = {}
            for k in knowledge_list:
                # PDF 파일 수
                pdf_files = KnowledgeModel.list_pdf_files(k.id)
                pdf_count = len(pdf_files) if pdf_files else 0
                
                # 이미지 파일 수
                image_files = KnowledgeModel.list_images(k.id)
                image_count = len(image_files) if image_files else 0
                
                # 임베딩 통계
                from app.models.embedding import EmbeddingManager
                embedding_manager = EmbeddingManager()
                embedding_stats = embedding_manager.get_embedding_stats(
                    KnowledgeModel.get_embedding_path(k.id)
                )
                embedding_count = embedding_stats.get("total_embeddings", 0)
                
                # 표시 이름 (메타데이터 포함)
                display_name = f"{k.name} (PDF: {pdf_count}, 이미지: {image_count}, 임베딩: {embedding_count})"
                knowledge_options[k.id] = display_name
            
            # 기본 선택 인덱스 설정
            default_index = 0
            if "selected_knowledge_id" in st.session_state:
                selected_id = st.session_state["selected_knowledge_id"]
                ids = list(knowledge_options.keys())
                if selected_id in ids:
                    default_index = ids.index(selected_id)
            
            # Knowledge 선택 드롭다운
            selected_id = st.selectbox(
                "Knowledge를 선택하세요", 
                options=list(knowledge_options.keys()),
                format_func=lambda x: knowledge_options[x],
                index=default_index
            )
            
            # 선택된 Knowledge ID 설정
            if selected_id:
                st.session_state["selected_knowledge_id"] = selected_id
                # 선택된 Knowledge 정보 찾기
                selected_knowledge = next((k for k in knowledge_list if k.id == selected_id), None)
                
                if selected_knowledge:
                    # Knowledge 정보 표시
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        
                        # PDF 파일 수
                        pdf_files = KnowledgeModel.list_pdf_files(selected_id)
                        pdf_count = len(pdf_files) if pdf_files else 0
                        col1.metric("PDF", pdf_count)
                        
                        # 이미지 파일 수
                        image_files = KnowledgeModel.list_images(selected_id)
                        image_count = len(image_files) if image_files else 0
                        col2.metric("이미지", image_count)
                        
                        # 임베딩 수
                        from app.models.embedding import EmbeddingManager
                        embedding_manager = EmbeddingManager()
                        embedding_stats = embedding_manager.get_embedding_stats(
                            KnowledgeModel.get_embedding_path(selected_id)
                        )
                        embedding_count = embedding_stats.get("total_embeddings", 0)
                        col3.metric("임베딩", embedding_count)
                    
                    # Knowledge 정보 표시
                    with st.expander("Knowledge 정보", expanded=False):
                        st.markdown(f"**이름**: {selected_knowledge.name}")
                        if selected_knowledge.description:
                            st.markdown(f"**설명**: {selected_knowledge.description}")
                        st.markdown(f"**ID**: {selected_knowledge.id}")
                        st.markdown(f"**생성일**: {selected_knowledge.created_at.strftime('%Y-%m-%d %H:%M')}")
                        st.markdown(f"**마지막 업데이트**: {selected_knowledge.updated_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Knowledge 관리 버튼
                    col1, col2 = st.columns(2)
                    
                    # 통계 새로고침 버튼
                    if col1.button("통계 새로고침", use_container_width=True):
                        with st.spinner("통계 정보 새로고침 중..."):
                            # 파일 개수 기반으로 통계 업데이트
                            KnowledgeModel.update_knowledge_status(
                                selected_id,
                                pdf_count=pdf_count,
                                image_count=image_count,
                                embedding_count=embedding_count
                            )
                            st.success("통계 정보가 갱신되었습니다.")
                            st.rerun()
                    
                    # 삭제 버튼
                    if col2.button("삭제", use_container_width=True):
                        with st.popover("정말 삭제하시겠습니까?", use_container_width=True):
                            if st.button("예, 삭제합니다", type="primary", use_container_width=True):
                                with st.spinner("Knowledge 삭제 중..."):
                                    KnowledgeModel.delete_knowledge(selected_id)
                                    st.success(f"Knowledge '{selected_knowledge.name}'가 삭제되었습니다.")
                                    # 세션 상태 초기화
                                    if "selected_knowledge_id" in st.session_state:
                                        del st.session_state["selected_knowledge_id"]
                                    st.rerun()
        
        st.divider()
        
        # 시스템 정보
        with st.expander("시스템 정보", expanded=False):
            # Torch 디바이스 확인
            device_type = "CPU"
            if torch.cuda.is_available():
                device_type = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
            elif torch.backends.mps.is_available():
                device_type = "Apple Silicon (MPS)"
            
            st.markdown(f"**모델**: {MODEL_NAME}")
            st.markdown(f"**디바이스**: {device_type}")
            st.markdown(f"**Python 버전**: {sys.version.split()[0]}")
            st.markdown(f"**Torch 버전**: {torch.__version__}")
            st.markdown(f"**Multimodal RAG v1.0**")

def handle_pdf_upload(knowledge_id):
    """
    PDF 파일 업로드 및 처리
    
    Args:
        knowledge_id: Knowledge ID
    """
    if not knowledge_id:
        st.warning("먼저 사이드바에서 Knowledge를 선택하거나 생성해주세요.")
        return
    
    st.header("PDF 파일 업로드")
    
    # 파일 업로드 위젯
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)}개의 파일이 업로드되었습니다.")
        
        if st.button("업로드한 PDF 처리하기"):
            with st.spinner("PDF 저장 및 처리 중..."):
                # 각 PDF 파일 처리
                for uploaded_file in uploaded_files:
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Knowledge에 PDF 저장
                        pdf_path = KnowledgeModel.save_pdf_file(
                            knowledge_id=knowledge_id,
                            pdf_file_path=tmp_file_path,
                            new_filename=uploaded_file.name
                        )
                        
                        if pdf_path:
                            st.success(f"PDF 파일 '{uploaded_file.name}'이 저장되었습니다.")
                        else:
                            st.error(f"PDF 파일 '{uploaded_file.name}' 저장 중 오류가 발생했습니다.")
                    
                    finally:
                        # 임시 파일 삭제
                        os.remove(tmp_file_path)
                
                # PDF 파일 목록 새로고침
                st.rerun()

def show_pdf_files(knowledge_id):
    """
    Knowledge에 저장된 PDF 파일 목록 표시
    
    Args:
        knowledge_id: Knowledge ID
    """
    if not knowledge_id:
        return
    
    st.header("저장된 PDF 파일")
    
    # PDF 파일 목록 조회
    pdf_files = KnowledgeModel.list_pdf_files(knowledge_id)
    
    if not pdf_files:
        st.info("저장된 PDF 파일이 없습니다.")
        return
    
    # PDF 파일 목록 DataFrame으로 변환하여 표시
    pdf_data = []
    for i, pdf_file in enumerate(pdf_files):
        file_stat = pdf_file.stat()
        pdf_data.append({
            "번호": i + 1,
            "파일명": pdf_file.name,
            "크기": f"{file_stat.st_size / 1024:.1f} KB",
            "수정일": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    
    df = pd.DataFrame(pdf_data)
    st.dataframe(df, use_container_width=True)
    
    # PDF 변환 기능
    st.subheader("PDF를 이미지로 변환")
    
    # 변환 설정
    col1, col2 = st.columns(2)
    
    with col1:
        dpi = st.slider("이미지 해상도 (DPI)", min_value=72, max_value=600, value=DEFAULT_DPI, step=1,
                      help="높은 DPI는 더 좋은 화질이지만 파일 크기가 커지고 처리 시간이 길어집니다.")
    
    with col2:
        if st.button("모든 PDF 변환하기", type="primary", use_container_width=True):
            # 진행 상태 표시 컨테이너
            progress_container = st.container()
            
            # 전체 진행률
            overall_progress = progress_container.progress(0)
            
            # 파일 정보
            file_info = progress_container.empty()
            
            # 현재 파일 진행률
            file_progress = progress_container.progress(0)
            
            # 상태 텍스트
            status_text = progress_container.empty()
            
            # 진행 상태 업데이트 함수
            def update_progress(current_file, current_page, total_pages, file_index, total_files):
                # 파일 정보 업데이트
                file_info.markdown(f"**처리 중인 파일:** {current_file} ({file_index+1}/{total_files})")
                
                # 현재 파일 진행률 업데이트
                if total_pages > 0:
                    file_progress.progress(current_page / total_pages)
                    status_text.markdown(f"**페이지 진행률:** {current_page}/{total_pages} 페이지")
                else:
                    file_progress.progress(0.0)
                    status_text.markdown(f"**페이지 진행률:** 0/{total_pages} 페이지 (경고: 페이지 수가 0입니다)")
                
                # 전체 진행률 업데이트 (ZeroDivisionError 방지)
                if total_pages == 0 or total_files == 0:
                    overall_progress.progress(0.0)
                    st.warning(f"진행률 계산 중 0으로 나누기 발생: total_pages={total_pages}, total_files={total_files}. 파일을 확인하세요.")
                else:
                    overall_progress.progress((file_index + (current_page / total_pages)) / total_files)
            
            # 이미지 변환 실행
            with st.spinner("모든 PDF를 이미지로 변환 중..."):
                result = KnowledgeModel.convert_all_pdfs(knowledge_id, dpi=dpi, progress_callback=update_progress)
                
                # 결과 요약
                total_images = sum(len(images) for images in result.values())
                st.success(f"{len(result)}개의 PDF 파일로부터 총 {total_images}개의 이미지가 생성되었습니다.")
                
                # 통계 업데이트
                image_count = len(KnowledgeModel.list_images(knowledge_id))
                KnowledgeModel.update_knowledge_status(knowledge_id, image_count=image_count)
                
                # 페이지 재로드 버튼
                if st.button("새로고침", key="pdf_converted_refresh"):
                    st.rerun()

def show_images(knowledge_id):
    """
    Knowledge에 저장된 이미지 파일 목록 표시
    
    Args:
        knowledge_id: Knowledge ID
    """
    if not knowledge_id:
        return
    
    st.header("변환된 이미지")
    
    # 이미지 파일 목록 조회
    image_files = KnowledgeModel.list_images(knowledge_id)
    
    if not image_files:
        st.info("변환된 이미지 파일이 없습니다.")
        return
    
    # 이미지 개수 표시
    st.write(f"총 {len(image_files)}개의 이미지가 있습니다.")
    
    # 이미지 갤러리 또는 데이터프레임 선택
    view_type = st.radio("보기 방식", ["갤러리", "목록"], horizontal=True)
    
    if view_type == "갤러리":
        # 이미지 수가 많을 경우 샘플링
        max_display = st.slider("표시할 이미지 수", min_value=5, max_value=100, value=20, step=5)
        
        if len(image_files) > max_display:
            st.warning(f"이미지가 너무 많아 처음 {max_display}개만 표시합니다.")
            image_files = image_files[:max_display]
        
        # 이미지 갤러리 표시 (3열)
        cols = st.columns(3)
        for i, image_file in enumerate(image_files):
            with cols[i % 3]:
                try:
                    st.image(str(image_file), caption=image_file.name, use_column_width=True)
                except Exception as e:
                    st.error(f"이미지 로드 중 오류: {e}")
    else:
        # 목록 보기 (데이터프레임)
        max_rows = st.slider("표시할 행 수", min_value=10, max_value=100, value=25, step=5)
        
        # 이미지 파일 정보 추출
        image_data = []
        for i, image_file in enumerate(image_files[:max_rows]):
            file_stat = image_file.stat()
            
            # 파일명에서 정보 추출
            from app.utils.pdf import extract_page_info_from_filename
            page_info = extract_page_info_from_filename(image_file.stem)
            doc_id = page_info[0] if page_info else "N/A"
            page_num = page_info[1] if page_info else "N/A"
            
            image_data.append({
                "번호": i + 1,
                "파일명": image_file.name,
                "문서 ID": doc_id,
                "페이지": page_num,
                "크기": f"{file_stat.st_size / 1024:.1f} KB",
                "수정일": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        
        df = pd.DataFrame(image_data)
        st.dataframe(df, use_container_width=True)
        
        if len(image_files) > max_rows:
            st.info(f"총 {len(image_files)}개 중 {max_rows}개만 표시됩니다.")
    
    # 임베딩 생성 기능
    st.subheader("이미지 임베딩")
    
    # 모듈 설치 확인
    try:
        import importlib.util
        colpali_installed = importlib.util.find_spec("colpali_engine") is not None
    except:
        colpali_installed = False
    
    if not colpali_installed:
        st.warning("⚠️ 임베딩 모듈(colpali_engine)이 설치되지 않았습니다.")
        st.info("""
        임베딩 기능을 사용하려면 다음 단계를 따르세요:
        1. 터미널에서 `pip install colpali-engine` 명령으로 패키지를 설치하세요.
        2. 설치가 완료되면 앱을 재시작하세요.
        
        **참고**: 임베딩 모듈 없이도 PDF 업로드와 이미지 변환 기능은 계속 사용할 수 있습니다.
        """)
        
        if st.button("임베딩 생성 (모듈 없음)"):
            st.error("임베딩 모듈이 설치되지 않아 이 기능을 사용할 수 없습니다.")
            
    else:
        # 임베딩 설정
        col1, col2 = st.columns(2)
        
        with col1:
            # 배치 크기 설정
            batch_size = st.slider("배치 크기", min_value=1, max_value=32, value=4, step=1, 
                                help="한 번에 처리할 이미지 수입니다. GPU 메모리에 따라 조정하세요.")
            
        with col2:
            # 저메모리 모드
            low_memory_mode = st.checkbox("저메모리 모드", value=True,
                                       help="저사양 시스템에서 메모리 사용량을 최소화합니다. 처리 속도가 느려질 수 있습니다.")
        
        # 중간 저장 간격
        save_interval = st.slider("중간 저장 간격", min_value=10, max_value=200, value=50, step=10,
                               help="지정한 개수만큼 이미지가 처리될 때마다 중간 결과를 저장합니다.")
        
        # 이미지 최대 크기
        max_image_size = st.slider("이미지 최대 크기(px)", min_value=512, max_value=2048, value=1024, step=128,
                                help="큰 이미지는 지정한 크기로 자동 리사이즈됩니다. 메모리 사용량 감소에 도움됩니다.")
        
        # 기기 정보 표시
        device_info = "CPU"
        if torch.cuda.is_available():
            device_info = f"GPU ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            device_info = "Apple Silicon (MPS)"
            
        st.info(f"임베딩 생성에 사용할 기기: {device_info}")
        
        if st.button("임베딩 생성", type="primary", use_container_width=True):
            # 이미지 파일 수 확인
            image_files = KnowledgeModel.list_images(knowledge_id)
            if not image_files:
                st.warning("변환된 이미지 파일이 없습니다. 먼저 PDF를 이미지로 변환해주세요.")
            else:
                # 진행 상태 표시 컨테이너
                progress_container = st.container()
                
                # 상태 텍스트
                status_text = progress_container.empty()
                status_text.markdown("임베딩 생성 준비 중...")
                
                # 진행률
                progress_bar = progress_container.progress(0)
                
                # 진행 상태 업데이트 함수
                def update_embedding_progress(current, total, message):
                    progress = current / max(1, total)
                    progress_bar.progress(progress)
                    status_text.markdown(f"**상태:** {message}")
                
                with st.spinner("이미지 임베딩 생성 중... (시간이 오래 걸릴 수 있습니다)"):
                    try:
                        # 임베딩 생성
                        success = KnowledgeModel.generate_embeddings(
                            knowledge_id, 
                            batch_size=batch_size,
                            progress_callback=update_embedding_progress,
                            max_image_size=max_image_size,
                            save_interval=save_interval,
                            low_memory_mode=low_memory_mode
                        )
                        
                        if success:
                            # 임베딩 통계 조회
                            from app.models.embedding import EmbeddingManager
                            embedding_manager = EmbeddingManager()
                            embedding_stats = embedding_manager.get_embedding_stats(
                                KnowledgeModel.get_embedding_path(knowledge_id)
                            )
                            
                            # 통계 업데이트
                            KnowledgeModel.update_knowledge_status(
                                knowledge_id, 
                                embedding_count=embedding_stats.get("total_embeddings", 0)
                            )
                            
                            st.success(f"이미지 임베딩이 성공적으로 생성되었습니다. (총 {embedding_stats.get('total_embeddings', 0)}개)")
                        else:
                            st.error("이미지 임베딩 생성 중 오류가 발생했습니다.")
                    except Exception as e:
                        st.error(f"임베딩 생성 중 오류 발생: {str(e)}")
                        st.info("모델 로딩 또는 처리 중 문제가 발생했습니다. 콘솔 로그를 확인하세요.")

def main():
    """
    메인 애플리케이션 진입점
    """
    # 페이지 설정
    st.set_page_config(
        page_title="Multimodal RAG",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # 타이틀
    st.title("🔍 Multimodal RAG 시스템")
    
    # 기본 설명
    st.markdown("""
        여러 개의 문서를 입력받아 이미지로 변환한 뒤, 이미지 임베딩 벡터로 변환하여 저장하고 
        사용자의 질문에 맞는 이미지를 검색해주는 멀티모달 RAG 시스템입니다.
    """)
    
    # Knowledge 디렉토리 생성
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 사이드바 - Knowledge 관리
    create_knowledge_sidebar()
    
    # 선택된 Knowledge ID
    knowledge_id = st.session_state.get("selected_knowledge_id")
    
    # 첫번째 탭: Knowledge 정보
    tab1, tab2, tab3 = st.tabs(["📋 Knowledge 정보", "📄 파일 업로드 및 처리", "🔎 검색 및 평가"])
    
    with tab1:
        # Knowledge 정보 표시
        render_knowledge_view(knowledge_id)
    
    with tab2:
        # PDF 업로드 및 처리
        handle_pdf_upload(knowledge_id)
        
        # 저장된 PDF 파일 표시
        show_pdf_files(knowledge_id)
        
        # 변환된 이미지 표시
        show_images(knowledge_id)
    
    with tab3:
        # 검색 UI 표시
        render_search_ui(knowledge_id)

if __name__ == "__main__":
    main() 