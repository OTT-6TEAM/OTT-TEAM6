# 드라마 Overview 텍스트 임베딩 가이드

## 📦 필요한 패키지 설치

```bash
# 필수 패키지 설치
pip install sentence-transformers pandas pyarrow tqdm numpy

# 또는 requirements.txt로 설치
pip install -r requirements.txt
```

## 🚀 사용 방법

### 방법 1: 간단한 스크립트 실행 (추천)

```bash
python drama_embedding_simple.py
```

이 방법이 가장 간단하고 직관적입니다!

### 방법 2: 고급 기능 포함 스크립트

```bash
python drama_embedding.py
```

이 방법은 다음 기능을 포함합니다:
- 중간 저장 체크포인트 (매 100 배치마다)
- 더 자세한 로깅
- 검증 기능

### 방법 3: Jupyter Notebook에서 직접 실행

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. 데이터 로드
drama_final = pd.read_parquet("최종데이터셋_드라마/drama_final.parquet")

# 2. 모델 로드
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 3. 임베딩 생성
texts = drama_final['overview'].fillna("").tolist()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# 4. 데이터프레임에 추가
drama_final['embedding'] = list(embeddings)

# 5. 저장
drama_final.to_csv("drama_embeddings_progress.csv", index=False)
drama_final.to_parquet("drama_with_embeddings.parquet", index=False)

print("완료!")
```

## 📊 출력 파일

1. **drama_embeddings_progress.csv** - 중간 결과 (CSV 형식)
2. **drama_with_embeddings.parquet** - 최종 결과 (Parquet 형식, 압축됨)

## ⚙️ 설정 옵션

### batch_size 조정
- **GPU가 있는 경우**: 64 또는 128로 증가 → 더 빠름
- **메모리가 부족한 경우**: 16 또는 8로 감소 → 느리지만 안정적

```python
# drama_embedding_simple.py에서 이 줄을 수정:
batch_size = 32  # 원하는 값으로 변경
```

## 🔍 결과 확인

```python
import pandas as pd

# Parquet 파일 읽기
df = pd.read_parquet("drama_with_embeddings.parquet")

# 기본 정보
print(f"데이터 shape: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")

# 임베딩 확인
print(f"임베딩 차원: {len(df.iloc[0]['embedding'])}")
print(f"첫 번째 임베딩 샘플: {df.iloc[0]['embedding'][:5]}")
```

## 📈 성능 팁

1. **GPU 사용** (가능한 경우)
   - CUDA가 설치되어 있으면 자동으로 GPU 사용
   - CPU보다 10-50배 빠름

2. **배치 크기 최적화**
   - GPU 메모리: 8GB → batch_size=64
   - GPU 메모리: 4GB → batch_size=32
   - CPU만 사용 → batch_size=16

3. **예상 소요 시간**
   - 1,000개 드라마: 약 1-2분 (GPU) / 5-10분 (CPU)
   - 10,000개 드라마: 약 10-20분 (GPU) / 50-100분 (CPU)

## 🛠️ 문제 해결

### 메모리 부족 에러
```python
# batch_size를 줄이세요
batch_size = 16  # 또는 8
```

### CUDA 에러
```python
# CPU 강제 사용
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### 모델 다운로드 느림
- 첫 실행시 모델 다운로드 (약 400MB)
- 이후에는 캐시 사용으로 빠름

## 📌 모델 정보

**sentence-transformers/all-mpnet-base-v2**
- 임베딩 차원: 768
- 최대 시퀀스 길이: 384 토큰
- 언어: 영어
- 용도: 문장/텍스트 유사도, 검색, 클러스터링

## 💡 활용 예제

### 1. 유사 드라마 찾기
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 임베딩 배열로 변환
embedding_matrix = np.vstack(df['embedding'].values)

# 첫 번째 드라마와 유사한 드라마 찾기
similarities = cosine_similarity([embedding_matrix[0]], embedding_matrix)[0]
top_5_idx = np.argsort(similarities)[-6:-1][::-1]  # 자기 자신 제외

print("유사한 드라마:")
for idx in top_5_idx:
    print(f"- {df.iloc[idx]['overview'][:100]}...")
    print(f"  유사도: {similarities[idx]:.3f}\n")
```

### 2. 드라마 클러스터링
```python
from sklearn.cluster import KMeans

# K-Means 클러스터링
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(embedding_matrix)

# 클러스터별 드라마 수
print(df['cluster'].value_counts())
```

### 3. 키워드로 드라마 검색
```python
# 검색 쿼리 임베딩
query = "crime investigation detective"
query_embedding = model.encode([query])[0]

# 유사도 계산
similarities = cosine_similarity([query_embedding], embedding_matrix)[0]
top_10_idx = np.argsort(similarities)[-10:][::-1]

print(f"'{query}' 검색 결과:")
for idx in top_10_idx:
    print(f"- {df.iloc[idx]['overview'][:100]}...")
    print(f"  유사도: {similarities[idx]:.3f}\n")
```
