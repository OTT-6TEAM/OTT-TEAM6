"""
Movie Overview + Genre 텍스트 임베딩 코드
Qwen/Qwen3-Embedding-0.6B 모델 사용
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# =====================================================
# 1. 데이터 로드
# =====================================================
print("=" * 50)
print("1. 데이터 로드")
print("=" * 50)

movie_main = pd.read_parquet("최종데이터셋_영화/00_movie_main.parquet")
movie_genre = pd.read_parquet("최종데이터셋_영화/02_movie_genre.parquet")

print(f"movie_main shape: {movie_main.shape}")
print(f"movie_genre shape: {movie_genre.shape}")

# =====================================================
# 2. 장르 데이터 집계 (imdb_id별로 장르 합치기)
# =====================================================
print("\n" + "=" * 50)
print("2. 장르 데이터 집계 (imdb_id별로 장르 합치기)")
print("=" * 50)

# imdb_id별로 장르를 쉼표로 연결
# 예: tt0412175 -> "Drama, Documentary, Comedy"
genres_grouped = movie_genre.groupby('imdb_id')['genre'].apply(
    lambda x: ', '.join(x.dropna().astype(str).str.strip())
).reset_index()
genres_grouped.columns = ['imdb_id', 'genres_combined']

print(f"집계된 장르 데이터 shape: {genres_grouped.shape}")
print("\n장르 집계 예시:")
print(genres_grouped.head())

# =====================================================
# 3. 데이터 병합
# =====================================================
print("\n" + "=" * 50)
print("3. 데이터 병합 (imdb_id 기준)")
print("=" * 50)

# movie_main과 genres_grouped 병합
merged_df = movie_main.merge(genres_grouped, on='imdb_id', how='left')

# overview와 genres_combined를 합쳐서 텍스트 생성
def combine_text(row):
    genres = row.get('genres_combined', '')
    overview = row.get('overview', '')
    
    # None이나 NaN 처리
    if pd.isna(genres) or genres == '':
        genres = 'Unknown'
    if pd.isna(overview) or overview == '':
        overview = 'No overview available'
    
    # 텍스트 조합: "Genres: [장르]. Overview: [줄거리]"
    combined = f"Genres: {genres}. Overview: {overview}"
    return combined

merged_df['combined_text'] = merged_df.apply(combine_text, axis=1)

print(f"병합된 데이터 shape: {merged_df.shape}")
print("\n병합된 텍스트 예시:")
for i in range(min(3, len(merged_df))):
    print(f"\n[{i}] {merged_df['combined_text'].iloc[i][:200]}...")

# =====================================================
# 4. 중간 결과 CSV 저장
# =====================================================
print("\n" + "=" * 50)
print("4. 중간 결과 CSV 저장")
print("=" * 50)

intermediate_csv_path = "최종데이터셋_영화/movie_combined_text_intermediate.csv"
merged_df.to_csv(intermediate_csv_path, index=False, encoding='utf-8-sig')
print(f"중간 결과 저장 완료: {intermediate_csv_path}")

# =====================================================
# 5. 임베딩 모델 로드
# =====================================================
print("\n" + "=" * 50)
print("5. Qwen/Qwen3-Embedding-0.6B 모델 로드")
print("=" * 50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 디바이스: {device}")

model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device=device)
print("모델 로드 완료")
print(f"임베딩 차원: {model.get_sentence_embedding_dimension()}")

# =====================================================
# 6. 텍스트 임베딩 수행
# =====================================================
print("\n" + "=" * 50)
print("6. 텍스트 임베딩 수행")
print("=" * 50)

texts = merged_df['combined_text'].tolist()

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"\n임베딩 완료!")
print(f"임베딩 shape: {embeddings.shape}")

# =====================================================
# 7. 임베딩 결과를 데이터프레임에 추가
# =====================================================
print("\n" + "=" * 50)
print("7. 임베딩 결과 저장 준비")
print("=" * 50)

merged_df['embedding'] = [emb.tolist() for emb in embeddings]

print(f"최종 데이터프레임 shape: {merged_df.shape}")
print(f"컬럼 목록: {merged_df.columns.tolist()}")

# =====================================================
# 8. 최종 결과 Parquet 저장
# =====================================================
print("\n" + "=" * 50)
print("8. 최종 결과 Parquet 저장")
print("=" * 50)

final_parquet_path = "최종데이터셋_영화/movie_text_embedding_qwen3.parquet"
merged_df.to_parquet(final_parquet_path, index=False)
print(f"최종 결과 저장 완료: {final_parquet_path}")

# =====================================================
# 9. 검증
# =====================================================
print("\n" + "=" * 50)
print("9. 결과 검증")
print("=" * 50)

test_load = pd.read_parquet(final_parquet_path)
print(f"저장된 파일 shape: {test_load.shape}")
print(f"임베딩 차원 확인: {len(test_load['embedding'].iloc[0])}")
print(f"\n임베딩 샘플 (첫 5개 값): {test_load['embedding'].iloc[0][:5]}")

print("\n" + "=" * 50)
print("완료!")
print("=" * 50)
print(f"- 중간 결과 (CSV): {intermediate_csv_path}")
print(f"- 최종 결과 (Parquet): {final_parquet_path}")