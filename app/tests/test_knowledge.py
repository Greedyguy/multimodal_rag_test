"""
Knowledge 모델 테스트
"""
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import unittest

from app.models.knowledge import KnowledgeModel, KnowledgeInfo
from app.utils.constants import KNOWLEDGE_DIR

class TestKnowledgeModel(unittest.TestCase):
    """Knowledge 모델 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 임시 디렉토리를 테스트용 Knowledge 디렉토리로 사용
        self.temp_dir = tempfile.mkdtemp()
        self.original_base_path = KnowledgeModel.BASE_PATH
        KnowledgeModel.BASE_PATH = Path(self.temp_dir)
        
        # 테스트용 Knowledge 생성
        self.test_knowledge = KnowledgeModel.create_knowledge(
            name="테스트 지식",
            description="테스트용 지식 데이터"
        )
    
    def tearDown(self):
        """테스트 정리"""
        # 원래 BASE_PATH 복원
        KnowledgeModel.BASE_PATH = self.original_base_path
        
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_dir)
    
    def test_create_knowledge(self):
        """Knowledge 생성 테스트"""
        # Knowledge 정보 확인
        self.assertEqual(self.test_knowledge.name, "테스트 지식")
        self.assertEqual(self.test_knowledge.description, "테스트용 지식 데이터")
        self.assertEqual(self.test_knowledge.pdf_count, 0)
        self.assertEqual(self.test_knowledge.image_count, 0)
        self.assertEqual(self.test_knowledge.embedding_count, 0)
        
        # Knowledge 디렉토리 생성 확인
        knowledge_path = KnowledgeModel.get_knowledge_path(self.test_knowledge.id)
        self.assertTrue(knowledge_path.exists(), "Knowledge 디렉토리가 생성되지 않았습니다")
        
        # PDF, 이미지, 임베딩 디렉토리 생성 확인
        self.assertTrue(KnowledgeModel.get_pdf_path(self.test_knowledge.id).exists(), "PDF 디렉토리가 생성되지 않았습니다")
        self.assertTrue(KnowledgeModel.get_image_path(self.test_knowledge.id).exists(), "이미지 디렉토리가 생성되지 않았습니다")
        self.assertTrue(KnowledgeModel.get_embedding_path(self.test_knowledge.id).exists(), "임베딩 디렉토리가 생성되지 않았습니다")
    
    def test_list_knowledge_infos(self):
        """Knowledge 목록 조회 테스트"""
        # 추가 Knowledge 생성
        KnowledgeModel.create_knowledge(name="두 번째 지식", description="두 번째 테스트")
        
        # Knowledge 목록 조회
        knowledge_list = KnowledgeModel.list_knowledge_infos()
        
        # 생성한 Knowledge가 목록에 있는지 확인
        self.assertEqual(len(knowledge_list), 2, "Knowledge 목록 수가 일치하지 않습니다")
        
        # Knowledge 이름 확인
        knowledge_names = [k["name"] for k in knowledge_list]
        self.assertIn("테스트 지식", knowledge_names, "첫 번째 Knowledge가 목록에 없습니다")
        self.assertIn("두 번째 지식", knowledge_names, "두 번째 Knowledge가 목록에 없습니다")
    
    def test_get_knowledge_info(self):
        """Knowledge 정보 조회 테스트"""
        # Knowledge 정보 조회
        knowledge = KnowledgeModel.get_knowledge_info(self.test_knowledge.id)
        
        # Knowledge 정보 확인
        self.assertIsNotNone(knowledge, "Knowledge 정보를 찾을 수 없습니다")
        self.assertEqual(knowledge.name, "테스트 지식", "Knowledge 이름이 일치하지 않습니다")
        self.assertEqual(knowledge.description, "테스트용 지식 데이터", "Knowledge 설명이 일치하지 않습니다")
    
    def test_update_knowledge_status(self):
        """Knowledge 상태 업데이트 테스트"""
        # Knowledge 상태 업데이트
        KnowledgeModel.update_knowledge_status(
            self.test_knowledge.id,
            pdf_count=5,
            image_count=10,
            embedding_count=10
        )
        
        # 업데이트된 Knowledge 정보 조회
        updated_knowledge = KnowledgeModel.get_knowledge_info(self.test_knowledge.id)
        
        # 업데이트된 상태 확인
        self.assertEqual(updated_knowledge.pdf_count, 5, "PDF 카운트가 업데이트되지 않았습니다")
        self.assertEqual(updated_knowledge.image_count, 10, "이미지 카운트가 업데이트되지 않았습니다")
        self.assertEqual(updated_knowledge.embedding_count, 10, "임베딩 카운트가 업데이트되지 않았습니다")
    
    def test_delete_knowledge(self):
        """Knowledge 삭제 테스트"""
        # Knowledge 삭제
        result = KnowledgeModel.delete_knowledge(self.test_knowledge.id)
        
        # 삭제 결과 확인
        self.assertTrue(result, "Knowledge 삭제에 실패했습니다")
        
        # Knowledge 목록에서 삭제되었는지 확인
        knowledge_list = KnowledgeModel.list_knowledge_infos()
        knowledge_ids = [k["id"] for k in knowledge_list]
        self.assertNotIn(self.test_knowledge.id, knowledge_ids, "Knowledge가 목록에서 삭제되지 않았습니다")
    
    def test_search_and_filter(self):
        """검색 및 필터링 테스트"""
        # 추가 Knowledge 생성
        KnowledgeModel.create_knowledge(name="검색 테스트", description="검색용 Knowledge")
        KnowledgeModel.create_knowledge(name="필터 테스트", description="필터용 Knowledge")
        
        # 첫 번째 Knowledge 상태 업데이트
        KnowledgeModel.update_knowledge_status(self.test_knowledge.id, pdf_count=5)
        
        # 이름으로 검색
        search_results = KnowledgeModel.search_knowledge_by_name("테스트")
        self.assertEqual(len(search_results), 2, "이름 검색 결과가 일치하지 않습니다")
        
        # 파일 수로 필터링
        filtered_results = KnowledgeModel.filter_knowledge_by_criteria(min_pdfs=3)
        self.assertEqual(len(filtered_results), 1, "PDF 수 필터링 결과가 일치하지 않습니다")
        self.assertEqual(filtered_results[0].id, self.test_knowledge.id, "PDF 수 필터링 결과 ID가 일치하지 않습니다")
        
        # 파일 유형으로 필터링
        type_results = KnowledgeModel.get_knowledge_by_file_count("pdf", min_count=3)
        self.assertEqual(len(type_results), 1, "파일 유형 필터링 결과가 일치하지 않습니다")
        
        # 정렬 테스트
        sorted_results = KnowledgeModel.sort_knowledge_by_field("pdf_count", ascending=False)
        self.assertEqual(sorted_results[0].id, self.test_knowledge.id, "정렬 결과가 일치하지 않습니다")

if __name__ == "__main__":
    unittest.main() 