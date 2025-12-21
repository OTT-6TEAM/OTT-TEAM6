"""
영화 흥행/비흥행 BERTopic 유사 토픽 통합 분석
- 기존 BERTopic 분석 결과를 로드하여 유사 토픽을 클러스터링
- 흥행작: n_groups=8, 비흥행작: n_groups=6
- 결과를 'BERTOPIC_SIMP_MOVIE' 폴더에 저장
"""

import pandas as pd
import numpy as np
import os
from bertopic import BERTopic
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 0. 입출력 폴더 설정
# ==========================================================
INPUT_DIR = "영화데이터BERTOPIC"
OUTPUT_DIR = "BERTOPIC_SIMP_MOVIE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"입력 폴더: {INPUT_DIR}/")
print(f"출력 폴더 생성: {OUTPUT_DIR}/")

# ==========================================================
# 1. BERTopic 모델 로드
# ==========================================================
print("\n" + "="*60)
print("BERTopic 모델 로드 중...")
print("="*60)

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

hit_topic_model = BERTopic.load(
    f"{INPUT_DIR}/hit_bertopic_model",
    embedding_model=embedding_model
)
print(" ✓ 흥행작 BERTopic 모델 로드 완료")

flop_topic_model = BERTopic.load(
    f"{INPUT_DIR}/flop_bertopic_model",
    embedding_model=embedding_model
)
print(" ✓ 비흥행작 BERTopic 모델 로드 완료")

# ==========================================================
# 2. 영화 토픽 데이터 로드
# ==========================================================
df_hit = pd.read_csv(f"{INPUT_DIR}/hit_movie_topics.csv")
df_flop = pd.read_csv(f"{INPUT_DIR}/flop_movie_topics.csv")

print(f" ✓ 흥행작 영화 수: {len(df_hit)}")
print(f" ✓ 비흥행작 영화 수: {len(df_flop)}")

# ==========================================================
# 3. 유사 토픽 클러스터링 함수
# ==========================================================
def analyze_topic_clusters(topic_model, n_groups, label=""):
    topic_embeddings = topic_model.topic_embeddings_

    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()

    valid_embeddings = topic_embeddings[1:len(valid_topics)+1]

    actual_n_groups = min(n_groups, len(valid_topics))

    clustering = AgglomerativeClustering(
        n_clusters=actual_n_groups,
        metric="cosine",
        linkage="average"
    )
    cluster_labels = clustering.fit_predict(valid_embeddings)

    topic_clusters = pd.DataFrame({
        "topic_id": valid_topics,
        "cluster": cluster_labels,
        "movie_count": [
            topic_info[topic_info["Topic"] == t]["Count"].values[0]
            for t in valid_topics
        ],
        "keywords": [
            ", ".join([w for w, _ in topic_model.get_topic(t)[:5]])
            for t in valid_topics
        ]
    })

    print(f"\n📊 {label} 유사 토픽 통합 결과")
    for cid in sorted(topic_clusters["cluster"].unique()):
        subset = topic_clusters[topic_clusters["cluster"] == cid]
        print(f"\n📌 그룹 {cid} | 토픽 {subset['topic_id'].tolist()} | 총 {subset['movie_count'].sum()}편")
        for _, r in subset.iterrows():
            print(f"   Topic {r.topic_id}: {r.keywords}")

    summary = topic_clusters.groupby("cluster").agg({
        "topic_id": lambda x: list(x),
        "movie_count": "sum"
    }).reset_index()

    return topic_clusters, summary

# ==========================================================
# 4. 흥행작 토픽 통합 (n_groups=8)
# ==========================================================
hit_clusters, hit_summary = analyze_topic_clusters(
    hit_topic_model,
    n_groups=8,
    label="흥행작"
)

# ==========================================================
# 5. 비흥행작 토픽 통합 (n_groups=6)
# ==========================================================
flop_clusters, flop_summary = analyze_topic_clusters(
    flop_topic_model,
    n_groups=6,
    label="비흥행작"
)

# ==========================================================
# 6. 결과 저장
# ==========================================================
hit_clusters.to_csv(f"{OUTPUT_DIR}/hit_topic_clusters.csv", index=False, encoding="utf-8-sig")
flop_clusters.to_csv(f"{OUTPUT_DIR}/flop_topic_clusters.csv", index=False, encoding="utf-8-sig")

hit_summary.to_csv(f"{OUTPUT_DIR}/hit_cluster_summary.csv", index=False, encoding="utf-8-sig")
flop_summary.to_csv(f"{OUTPUT_DIR}/flop_cluster_summary.csv", index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print(f"유사 토픽 통합 완료! 결과 저장 위치: {OUTPUT_DIR}/")
print("="*60)
