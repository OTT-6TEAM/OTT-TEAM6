import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 모델 로드
print("Loading sentence-transformers model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully!")

# 데이터 로드
print("\nLoading data...")
drama_keyword_final = pd.read_parquet("최종데이터셋_드라마/drama_keyword_final.parquet")
print(f"Data loaded: {len(drama_keyword_final)} rows")

# keyword 컬럼 확인
print(f"\nKeyword column info:")
print(f"Total keywords: {len(drama_keyword_final)}")
print(f"Null values: {drama_keyword_final['keyword'].isna().sum()}")

# 결측값 처리 (있을 경우 빈 문자열로 대체)
drama_keyword_final['keyword'] = drama_keyword_final['keyword'].fillna('')

# 텍스트 임베딩 생성 (배치 처리로 속도 향상)
print("\nGenerating embeddings...")
batch_size = 256  # 배치 크기 (메모리에 따라 조정 가능)

keywords_list = drama_keyword_final['keyword'].tolist()

# 배치 처리로 임베딩 생성
embeddings = model.encode(
    keywords_list,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# 임베딩을 데이터프레임에 추가
drama_keyword_final['keyword_embedding'] = embeddings.tolist()

# 중간 결과를 CSV로 저장 (임베딩은 리스트 형태로 저장됨)
print("\nSaving intermediate results to CSV...")
drama_keyword_final.to_csv(
    "최종데이터셋_드라마/drama_keyword_embedded_intermediate.csv",
    index=False
)
print("Intermediate CSV saved!")

# 최종 결과를 Parquet으로 저장
print("\nSaving final results to Parquet...")
drama_keyword_final.to_parquet(
    "최종데이터셋_드라마/drama_keyword_embedded_final.parquet",
    index=False,
    compression='snappy'  # 압축으로 파일 크기 최적화
)
print("Final Parquet saved!")

# 결과 확인
print("\n=== Results Summary ===")
print(f"Total rows processed: {len(drama_keyword_final)}")
print(f"Embedding dimension: {len(drama_keyword_final['keyword_embedding'].iloc[0])}")
print(f"\nFirst 3 keywords:")
for i in range(min(3, len(drama_keyword_final))):
    print(f"{i}: {drama_keyword_final['keyword'].iloc[i]}")
print(f"\nFirst embedding (first 10 dimensions):")
print(drama_keyword_final['keyword_embedding'].iloc[0][:10])

print("\n✓ Process completed successfully!")
