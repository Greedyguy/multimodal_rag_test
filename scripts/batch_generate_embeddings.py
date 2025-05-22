   # tests/batch_generate_embeddings.py
   import sys
   from app.models.knowledge import KnowledgeModel

   if __name__ == "__main__":
       knowledge_id = sys.argv[1]  # 예: "my_knowledge_id"
       batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
       # "none" 또는 "0" 입력 시 None 처리
       max_image_size_arg = sys.argv[3] if len(sys.argv) > 3 else "1024"
       if max_image_size_arg.lower() == "none" or max_image_size_arg == "0":
           max_image_size = None
       else:
           max_image_size = int(max_image_size_arg)

       print(f"임베딩 생성 시작: knowledge_id={knowledge_id}, batch_size={batch_size}, max_image_size={max_image_size}")
       success = KnowledgeModel.generate_embeddings(
           knowledge_id,
           batch_size=batch_size,
           max_image_size=max_image_size,
           progress_callback=None,  # 콘솔 출력만
           save_interval=50,
           low_memory_mode=True
       )
       if success:
           print("임베딩 생성 완료!")
       else:
           print("임베딩 생성 실패!")
        # 실행 명령어   
        #    nohup env PYTHONPATH=. poetry run python tests/batch_generate_embeddings.py k_1747896371 4 3072 > embedding.log 2>&1 &