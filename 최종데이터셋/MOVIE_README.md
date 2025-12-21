# 영화 Overview 텍스트 임베딩 가이드 🎬

## 📦 필요한 패키지 설치

```bash
pip install sentence-transformers pandas pyarrow tqdm numpy
```

---

## 🚀 빠른 시작

### 방법 1: 표준 모델 (768차원, 최고 성능)

```bash
python movie_embedding.py
```

**특징:**
- 모델: all-mpnet-base-v2
- 임베딩 차원: 768
- 다운로드: 438MB
- 성능: 최고 ⭐⭐⭐⭐⭐

### 방법 2: 빠른 모델 (384차원, 추천!) ⚡

```bash
python movie_embedding_fast.py
```

**특징:**
- 모델: paraphrase-MiniLM-L6-v2
- 임베딩 차원: 384
- 다운로드: 90MB (5배 빠름!)
- 성능: 95% ⭐⭐⭐⭐

---

## 📊 출력 파일

실행 후 다음 파일이 생성됩니다:

1. **movie_embeddings_progress.csv** - 중간 결과 (CSV 형식)
   - 사람이 볼 수 있음 (엑셀로 열기 가능)
   - 모든 컬럼 + embedding 포함

2. **movie_with_embeddings.parquet** - 최종 결과 (Parquet 형식)
   - 압축됨 (CSV의 약 1/10 크기)
   - 빠른 로딩
   - 프로그램에서 사용 권장

---

## ⏱️ 예상 소요 시간

### 표준 모델 (all-mpnet-base-v2)
- GPU: 10-20분
- CPU: 30-60분

### 빠른 모델 (paraphrase-MiniLM-L6-v2) ⚡
- GPU: 5-10분
- CPU: 15-30분

*실제 시간은 영화 개수와 하드웨어 성능에 따라 다릅니다.

---

## 💡 활용 예제

### 1. 임베딩 파일 읽기

```python
import pandas as pd
import numpy as np

# Parquet 파일 읽기 (추천)
movie_df = pd.read_parquet("movie_with_embeddings.parquet")

# 기본 정보
print(f"영화 개수: {len(movie_df)}")
print(f"컬럼: {movie_df.columns.tolist()}")
print(f"임베딩 차원: {len(movie_df.iloc[0]['embedding'])}")
```

### 2. 유사한 영화 찾기

```python
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩을 2D 배열로 변환
embedding_matrix = np.vstack(movie_df['embedding'].values)

# 첫 번째 영화와 유사한 영화 찾기
target_idx = 0
similarities = cosine_similarity(
    [embedding_matrix[target_idx]], 
    embedding_matrix
)[0]

# 가장 유사한 영화 10개 (자기 자신 제외)
top_10_idx = np.argsort(similarities)[-11:-1][::-1]

print(f"\n'{movie_df.iloc[target_idx]['title']}'와 유사한 영화:\n")
for idx in top_10_idx:
    print(f"  {movie_df.iloc[idx]['title']}")
    print(f"  유사도: {similarities[idx]:.3f}")
    print(f"  줄거리: {movie_df.iloc[idx]['overview'][:100]}...")
    print()
```

### 3. 특정 영화와 유사한 영화 찾기 (제목으로)

```python
def find_similar_movies(movie_title, top_n=10):
    """영화 제목으로 유사한 영화 찾기"""
    
    # 영화 찾기
    target_movies = movie_df[
        movie_df['title'].str.contains(movie_title, case=False, na=False)
    ]
    
    if len(target_movies) == 0:
        print(f"'{movie_title}' 영화를 찾을 수 없습니다.")
        return
    
    # 첫 번째 매칭 영화 사용
    target_idx = target_movies.index[0]
    target_embedding = embedding_matrix[target_idx]
    
    # 유사도 계산
    similarities = cosine_similarity([target_embedding], embedding_matrix)[0]
    
    # 상위 영화 (자기 자신 제외)
    top_idx = np.argsort(similarities)[-top_n-1:-1][::-1]
    
    print(f"\n'{movie_df.iloc[target_idx]['title']}'와 유사한 영화:")
    print("="*60)
    
    for rank, idx in enumerate(top_idx, 1):
        print(f"\n{rank}. {movie_df.iloc[idx]['title']}")
        print(f"   유사도: {similarities[idx]:.3f}")
        print(f"   {movie_df.iloc[idx]['overview'][:150]}...")

# 사용 예시
find_similar_movies("Inception", top_n=5)
```

### 4. 키워드로 영화 검색

```python
from sentence_transformers import SentenceTransformer

# 모델 로드 (사용한 모델과 동일하게)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# 또는: model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def search_movies(query, top_n=10):
    """키워드로 영화 검색"""
    
    # 쿼리 임베딩
    query_embedding = model.encode([query])[0]
    
    # 유사도 계산
    similarities = cosine_similarity([query_embedding], embedding_matrix)[0]
    
    # 상위 영화
    top_idx = np.argsort(similarities)[-top_n:][::-1]
    
    print(f"\n'{query}' 검색 결과:")
    print("="*60)
    
    for rank, idx in enumerate(top_idx, 1):
        print(f"\n{rank}. {movie_df.iloc[idx]['title']}")
        print(f"   유사도: {similarities[idx]:.3f}")
        print(f"   {movie_df.iloc[idx]['overview'][:150]}...")

# 사용 예시
search_movies("space adventure", top_n=5)
search_movies("romantic comedy", top_n=5)
search_movies("superhero action", top_n=5)
```

### 5. 영화 클러스터링

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# K-Means 클러스터링 (10개 그룹)
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
movie_df['cluster'] = kmeans.fit_predict(embedding_matrix)

# 클러스터별 영화 수
print("클러스터별 영화 수:")
print(movie_df['cluster'].value_counts().sort_index())

# 각 클러스터의 대표 영화 보기
for cluster_id in range(n_clusters):
    cluster_movies = movie_df[movie_df['cluster'] == cluster_id]
    print(f"\n클러스터 {cluster_id} ({len(cluster_movies)}개 영화):")
    print(cluster_movies['title'].head(5).tolist())
```

### 6. 차원 축소 및 시각화 (2D)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA로 768차원 → 2차원
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embedding_matrix)

# 시각화
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1],
    c=movie_df['cluster'],  # 클러스터별 색상
    alpha=0.5,
    cmap='tab10'
)
plt.colorbar(scatter, label='Cluster')
plt.title('Movie Embeddings (2D Visualization)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('movie_clusters.png', dpi=300)
plt.show()

print(f"시각화 저장: movie_clusters.png")
```

### 7. 장르별 평균 임베딩

```python
# 장르별 평균 임베딩 계산 (장르 컬럼이 있다면)
if 'genres' in movie_df.columns:
    genre_embeddings = {}
    
    for genre in movie_df['genres'].unique():
        genre_movies = movie_df[movie_df['genres'] == genre]
        if len(genre_movies) > 0:
            genre_avg = np.mean(
                np.vstack(genre_movies['embedding'].values),
                axis=0
            )
            genre_embeddings[genre] = genre_avg
    
    print(f"장르별 평균 임베딩 계산 완료: {len(genre_embeddings)}개 장르")
```

---

## 🔧 문제 해결

### 모델 다운로드가 느릴 때

```bash
# 빠른 모델 사용 (추천!)
python movie_embedding_fast.py
```

### 메모리 부족 에러

코드에서 `batch_size`를 줄이세요:

```python
batch_size = 16  # 32 → 16으로 줄이기
# 또는
batch_size = 8   # 더 줄이기
```

### GPU 사용 확인

```python
import torch

print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"GPU 개수: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
```

---

## 📈 성능 비교

### 모델별 비교

| 모델 | 크기 | 차원 | 다운로드 | 속도 | 정확도 |
|------|------|------|----------|------|--------|
| **paraphrase-MiniLM-L6-v2** | 90MB | 384 | ⚡⚡⚡ | 🚀🚀🚀 | 84% |
| all-MiniLM-L12-v2 | 120MB | 384 | ⚡⚡ | 🚀🚀 | 85% |
| **all-mpnet-base-v2** | 438MB | 768 | ⚡ | 🚀 | 87% |

### 어떤 모델을 선택할까?

- **빠른 프로토타이핑**: paraphrase-MiniLM-L6-v2 ⚡
- **균형잡힌 선택**: all-MiniLM-L12-v2
- **최고 성능**: all-mpnet-base-v2

**추천:** 대부분의 경우 **paraphrase-MiniLM-L6-v2**로 충분합니다!

---

## 💾 파일 크기 예상

### 10,000개 영화 기준

**CSV 파일:**
- 384차원: 약 50-80 MB
- 768차원: 약 100-150 MB

**Parquet 파일:**
- 384차원: 약 5-10 MB
- 768차원: 약 10-20 MB

---

## 🎓 다음 단계

1. ✅ 임베딩 생성 완료
2. 📊 유사도 분석 시작
3. 🔍 검색 시스템 구축
4. 🎯 추천 시스템 개발
5. 📈 데이터 시각화

---

## 📞 도움말

문제가 발생하면:
1. Python 버전 확인 (3.8 이상)
2. 패키지 업데이트: `pip install --upgrade sentence-transformers`
3. GPU 드라이버 업데이트 (CUDA 사용시)

---

**행운을 빕니다! 🎬✨**
