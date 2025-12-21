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
movie_keyword = pd.read_parquet("최종데이터셋_영화/03_movie_keyword.parquet")
print(f"Data loaded: {len(movie_keyword)} rows")

# keywords 컬럼 확인
print(f"\nKeywords column info:")
print(f"Total keywords: {len(movie_keyword)}")
print(f"Null/None values: {movie_keyword['keywords'].isna().sum()}")
print(f"None as string: {(movie_keyword['keywords'] == 'None').sum()}")

# 결측값 처리 (None, NaN을 빈 문자열로 대체)
movie_keyword['keywords'] = movie_keyword['keywords'].fillna('')
movie_keyword['keywords'] = movie_keyword['keywords'].replace('None', '')

# 처리 후 확인
print(f"After cleaning - empty strings: {(movie_keyword['keywords'] == '').sum()}")

# 텍스트 임베딩 생성 (배치 처리로 속도 향상)
print("\nGenerating embeddings...")
batch_size = 256  # 배치 크기 (메모리에 따라 조정 가능)

keywords_list = movie_keyword['keywords'].tolist()

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
movie_keyword['keywords_embedding'] = embeddings.tolist()

# 중간 결과를 CSV로 저장 (임베딩은 리스트 형태로 저장됨)
print("\nSaving intermediate results to CSV...")
movie_keyword.to_csv(
    "최종데이터셋_영화/movie_keyword_embedded_intermediate.csv",
    index=False
)
print("Intermediate CSV saved!")

# 최종 결과를 Parquet으로 저장
print("\nSaving final results to Parquet...")
movie_keyword.to_parquet(
    "최종데이터셋_영화/movie_keyword_embedded_final.parquet",
    index=False,
    compression='snappy'  # 압축으로 파일 크기 최적화
)
print("Final Parquet saved!")

# 결과 확인
print("\n=== Results Summary ===")
print(f"Total rows processed: {len(movie_keyword)}")
print(f"Embedding dimension: {len(movie_keyword['keywords_embedding'].iloc[0])}")
print(f"\nFirst 5 keywords (non-empty):")
non_empty = movie_keyword[movie_keyword['keywords'] != '']['keywords'].head(5)
for i, keyword in enumerate(non_empty):
    print(f"{i}: {keyword}")

print(f"\nSample embedding (first 10 dimensions):")
# 비어있지 않은 첫 번째 키워드의 임베딩 출력
first_non_empty_idx = movie_keyword[movie_keyword['keywords'] != ''].index[0]
print(f"Keyword: '{movie_keyword['keywords'].iloc[first_non_empty_idx]}'")
print(movie_keyword['keywords_embedding'].iloc[first_non_empty_idx][:10])

print("\n✓ Process completed successfully!")
