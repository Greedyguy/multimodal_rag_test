import os
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm

# ColQwen2.5 모델 및 processor 임포트
from colpali_engine import ColQwen2_5, ColQwen2_5_Processor

# PDF → 이미지 변환 함수 (이미 구현된 함수 사용)
from app.utils.pdf import convert_pdf_to_images

# === 사용자 입력 ===
pdf_path = "/Users/luke/work/SKT/multimodal_rag/data/knowledge/k_1747392899/pdfs/(240411보도자료) 재정동향 4월호.pdf"  # 테스트할 PDF 파일 경로
output_image_dir = "/Users/luke/work/SKT/multimodal_rag/tmp_test_images"  # 임시 이미지 저장 폴더
query_text = "2024년 1월, 2월, 3월 각각의 평균 조달금리와 응찰률이 어떻게 되나요?"  # 테스트 쿼리
top_k = 3  # 상위 몇 개 결과 출력

# === ColQwen 모델 및 processor 로드 ===
model_name = "Metric-AI/ColQwen2.5-3b-multilingual"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"모델 로드 중... ({model_name}, device={device})")
model = ColQwen2_5.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
    device_map=device
).eval()
processor = ColQwen2_5_Processor.from_pretrained(model_name)
print("모델 및 processor 로드 완료")

# === PDF → 이미지 변환 ===
output_image_dir = Path(output_image_dir)
output_image_dir.mkdir(parents=True, exist_ok=True)
print(f"PDF → 이미지 변환 중... ({pdf_path})")
image_paths = convert_pdf_to_images(pdf_path, output_image_dir, dpi=300)
print(f"이미지 변환 완료: {len(image_paths)}개")

# === 이미지 임베딩 생성 ===
image_embeddings = []
page_nums = []
file_names = []

print("이미지 임베딩 생성 중...")
for img_path in tqdm(image_paths):
    img = Image.open(img_path).convert("RGB")
    processed_img = processor.process_images([img])
    processed_img = {k: v.to(device) for k, v in processed_img.items()}
    with torch.no_grad():
        emb = model(**processed_img)
    image_embeddings.append(emb)
    file_names.append(str(img_path))
    # 페이지 번호 추출 (파일명에서 숫자 추출)
    try:
        page_num = int(Path(img_path).stem.split("_")[-1])
    except Exception:
        page_num = 1
    page_nums.append(page_num)
print("이미지 임베딩 생성 완료")

# === 쿼리 임베딩 생성 ===
print(f"쿼리 임베딩 생성 중: '{query_text}'")
processed_query = processor.process_queries([query_text])
processed_query = {k: v.to(device) for k, v in processed_query.items()}
with torch.no_grad():
    query_emb = model(**processed_query)
print("쿼리 임베딩 생성 완료")

# === 유사도 계산 ===
print("유사도 계산 중...")
scores = score_multi_vector(query_emb, image_embeddings)
scores = scores[0].sort(descending=True)

print(f"\n=== Top {top_k} 결과 ===")
for i in range(min(top_k, len(scores.values))):
    idx = scores.indices[i]
    print(f"{i+1}. 파일명: {Path(file_names[idx]).name}, 페이지: {page_nums[idx]}, 유사도: {scores.values[i]:.4f}")

print("\n테스트 완료.") 