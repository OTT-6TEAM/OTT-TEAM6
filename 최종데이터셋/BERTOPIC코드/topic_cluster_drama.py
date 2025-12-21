"""
드라마 흥행/비흥행 BERTopic 유사 토픽 통합 분석
- 기존 BERTopic 분석 결과를 로드하여 유사 토픽을 클러스터링
- 흥행작: n_groups=4, 비흥행작: n_groups=8
- 결과를 'BERTOPIC_SIMP' 폴더에 저장
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
# 0. 출력 폴더 생성
# ==========================================================
INPUT_DIR = "드라마데이터BERTOPIC"
OUTPUT_DIR = "BERTOPIC_SIMP"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"입력 폴더: {INPUT_DIR}/")
print(f"출력 폴더 생성: {OUTPUT_DIR}/")

# ==========================================================
# 1. BERTopic 모델 및 데이터 로드
# ==========================================================
print("\n" + "="*60)
print("BERTopic 모델 및 데이터 로드 중...")
print("="*60)

# 임베딩 모델 로드 (모델 로드 시 필요)
embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

# BERTopic 모델 로드
hit_topic_model = BERTopic.load(
    f"{INPUT_DIR}/hit_bertopic_model",
    embedding_model=embedding_model
)
print("  ✓ 흥행작 BERTopic 모델 로드 완료")

flop_topic_model = BERTopic.load(
    f"{INPUT_DIR}/flop_bertopic_model",
    embedding_model=embedding_model
)
print("  ✓ 비흥행작 BERTopic 모델 로드 완료")

# 드라마 데이터 로드
df_hit = pd.read_csv(f"{INPUT_DIR}/hit_drama_topics.csv")
df_flop = pd.read_csv(f"{INPUT_DIR}/flop_drama_topics.csv")
print(f"  ✓ 흥행작 데이터: {len(df_hit)}개")
print(f"  ✓ 비흥행작 데이터: {len(df_flop)}개")

# ==========================================================
# 2. 토픽 클러스터 분석 함수
# ==========================================================

def analyze_topic_clusters(topic_model, n_groups, label=""):
    """
    토픽 간 거리를 계산하고 유사한 토픽끼리 그룹화
    
    Args:
        topic_model: 학습된 BERTopic 모델
        n_groups: 원하는 그룹 수
        label: 출력 시 표시할 라벨 (흥행작/비흥행작)
    
    Returns:
        topic_clusters: 토픽별 클러스터 정보 DataFrame
    """
    # 토픽 임베딩(좌표) 추출
    topic_embeddings = topic_model.topic_embeddings_
    
    # 토픽 정보 (outlier -1 제외)
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    
    # outlier(-1)는 인덱스 0에 있으므로, 실제 토픽은 인덱스 1부터
    valid_embeddings = topic_embeddings[1:len(valid_topics)+1]
    
    print(f"\n{'='*70}")
    print(f"📊 {label} 토픽 클러스터 분석")
    print(f"{'='*70}")
    print(f"토픽 수: {len(valid_topics)}개, 그룹 수: {n_groups}개")
    
    # 그룹 수가 토픽 수보다 많으면 조정
    actual_n_groups = min(n_groups, len(valid_topics))
    if actual_n_groups != n_groups:
        print(f"⚠️ 토픽 수({len(valid_topics)})가 그룹 수({n_groups})보다 적어 {actual_n_groups}개로 조정")
    
    # 계층적 클러스터링
    clustering = AgglomerativeClustering(
        n_clusters=actual_n_groups,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(valid_embeddings)
    
    # 결과 정리
    topic_clusters = pd.DataFrame({
        '토픽번호': valid_topics,
        '클러스터': cluster_labels,
        '드라마수': [topic_info[topic_info['Topic'] == t]['Count'].values[0] for t in valid_topics],
        '키워드': [', '.join([w for w, s in topic_model.get_topic(t)[:5]]) for t in valid_topics]
    })
    
    # 클러스터별 출력
    print(f"\n[유사 토픽 그룹]")
    
    for cluster_id in sorted(topic_clusters['클러스터'].unique()):
        cluster_topics = topic_clusters[topic_clusters['클러스터'] == cluster_id]
        topic_nums = cluster_topics['토픽번호'].tolist()
        total_dramas = cluster_topics['드라마수'].sum()
        
        print(f"\n📌 그룹 {cluster_id}: 토픽 {topic_nums} (총 {total_dramas}편)")
        print("-" * 60)
        
        for _, row in cluster_topics.iterrows():
            print(f"   토픽 {row['토픽번호']:2d} ({row['드라마수']:3d}편): {row['키워드']}")
    
    # 클러스터별 요약
    cluster_summary = topic_clusters.groupby('클러스터').agg({
        '토픽번호': lambda x: list(x),
        '드라마수': 'sum'
    }).reset_index()
    cluster_summary.columns = ['클러스터', '포함_토픽', '총_드라마수']
    
    print(f"\n[클러스터 요약]")
    print(cluster_summary.to_string(index=False))
    
    return topic_clusters, cluster_summary

# ==========================================================
# 3. 흥행작 토픽 클러스터 분석
# ==========================================================
print("\n" + "="*60)
print("흥행작 토픽 클러스터 분석")
print("="*60)

hit_topic_clusters, hit_cluster_summary = analyze_topic_clusters(
    topic_model=hit_topic_model,
    n_groups=4,  # 흥행작 4개 그룹
    label="흥행작"
)

# ==========================================================
# 4. 비흥행작 토픽 클러스터 분석
# ==========================================================
print("\n" + "="*60)
print("비흥행작 토픽 클러스터 분석")
print("="*60)

flop_topic_clusters, flop_cluster_summary = analyze_topic_clusters(
    topic_model=flop_topic_model,
    n_groups=8,  # 비흥행작 8개 그룹
    label="비흥행작"
)

# ==========================================================
# 5. 토픽 분석 요약 함수
# ==========================================================

def create_topic_summary(topic_model, df_subset, label):
    """토픽 분석 결과 요약"""
    topic_info = topic_model.get_topic_info()
    results = []
    
    for topic_id in sorted(topic_info['Topic'].unique()):
        if topic_id != -1:
            keywords = topic_model.get_topic(topic_id)
            top_keywords = [word for word, score in keywords[:5]]
            topic_dramas = df_subset[df_subset['topic'] == topic_id]
            
            # hit_score 평균 계산 (vote_average 대신)
            avg_hit_score = topic_dramas['hit_score'].mean() if len(topic_dramas) > 0 else 0
            
            results.append({
                'label': label,
                'topic_id': topic_id,
                'drama_count': len(topic_dramas),
                'avg_hit_score': round(avg_hit_score, 4) if not pd.isna(avg_hit_score) else 0,
                'keywords': ', '.join(top_keywords),
                'sample_dramas': ', '.join(topic_dramas['title'].head(3).tolist()) if 'title' in topic_dramas.columns else ''
            })
    
    return results

# ==========================================================
# 6. 키워드 차집합 분석
# ==========================================================
print("\n" + "="*60)
print("키워드 차집합 분석 중...")
print("="*60)

def get_all_keywords(topic_model):
    """모델의 모든 토픽에서 키워드와 점수 추출"""
    all_keywords = {}
    topic_info = topic_model.get_topic_info()
    
    for topic_id in topic_info['Topic'].values:
        if topic_id != -1:
            keywords = topic_model.get_topic(topic_id)
            for word, score in keywords:
                if word in all_keywords:
                    all_keywords[word] = max(all_keywords[word], score)
                else:
                    all_keywords[word] = score
    
    return all_keywords

# 흥행작/비흥행작 키워드 추출
hit_keywords = get_all_keywords(hit_topic_model)
flop_keywords = get_all_keywords(flop_topic_model)

# 차집합 계산
hit_unique_words = set(hit_keywords.keys()) - set(flop_keywords.keys())
flop_unique_words = set(flop_keywords.keys()) - set(hit_keywords.keys())

# 점수순 정렬
hit_unique_keywords = sorted([(w, hit_keywords[w]) for w in hit_unique_words], key=lambda x: -x[1])
flop_unique_keywords = sorted([(w, flop_keywords[w]) for w in flop_unique_words], key=lambda x: -x[1])

print(f"\n흥행작에만 있는 키워드 (상위 20개):")
for word, score in hit_unique_keywords[:20]:
    print(f"  {word}: {score:.4f}")

print(f"\n비흥행작에만 있는 키워드 (상위 20개):")
for word, score in flop_unique_keywords[:20]:
    print(f"  {word}: {score:.4f}")

# ==========================================================
# 7. 결과 저장
# ==========================================================
print("\n" + "="*60)
print("결과 저장 중...")
print("="*60)

# 1) 토픽 분석 요약
hit_summary = create_topic_summary(hit_topic_model, df_hit, 'hit')
flop_summary = create_topic_summary(flop_topic_model, df_flop, 'flop')
summary_df = pd.DataFrame(hit_summary + flop_summary)
summary_df.to_csv(f'{OUTPUT_DIR}/topic_analysis_summary.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/topic_analysis_summary.csv")

# 2) 드라마별 토픽 데이터
df_hit.to_csv(f'{OUTPUT_DIR}/hit_dramas_with_topics.csv', index=False, encoding='utf-8-sig')
df_flop.to_csv(f'{OUTPUT_DIR}/flop_dramas_with_topics.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/hit_dramas_with_topics.csv")
print(f"  ✓ {OUTPUT_DIR}/flop_dramas_with_topics.csv")

# 3) 토픽 클러스터 결과
hit_topic_clusters.to_csv(f'{OUTPUT_DIR}/hit_topic_clusters.csv', index=False, encoding='utf-8-sig')
flop_topic_clusters.to_csv(f'{OUTPUT_DIR}/flop_topic_clusters.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/hit_topic_clusters.csv")
print(f"  ✓ {OUTPUT_DIR}/flop_topic_clusters.csv")

# 4) 클러스터 요약
hit_cluster_summary.to_csv(f'{OUTPUT_DIR}/hit_cluster_summary.csv', index=False, encoding='utf-8-sig')
flop_cluster_summary.to_csv(f'{OUTPUT_DIR}/flop_cluster_summary.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/hit_cluster_summary.csv")
print(f"  ✓ {OUTPUT_DIR}/flop_cluster_summary.csv")

# 5) 키워드 비교 저장 (차집합 포함)
max_len = max(20, len(hit_unique_keywords), len(flop_unique_keywords))

# 리스트 길이 맞추기
hit_words = [w for w, s in hit_unique_keywords[:max_len]] + [''] * (max_len - min(max_len, len(hit_unique_keywords)))
hit_scores = [round(s, 4) for w, s in hit_unique_keywords[:max_len]] + [None] * (max_len - min(max_len, len(hit_unique_keywords)))
flop_words = [w for w, s in flop_unique_keywords[:max_len]] + [''] * (max_len - min(max_len, len(flop_unique_keywords)))
flop_scores = [round(s, 4) for w, s in flop_unique_keywords[:max_len]] + [None] * (max_len - min(max_len, len(flop_unique_keywords)))

keyword_comparison = pd.DataFrame({
    'hit_unique_keyword': hit_words[:max_len],
    'hit_unique_score': hit_scores[:max_len],
    'flop_unique_keyword': flop_words[:max_len],
    'flop_unique_score': flop_scores[:max_len],
})
keyword_comparison.to_csv(f'{OUTPUT_DIR}/keyword_comparison.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/keyword_comparison.csv")

# ==========================================================
# 8. 분석 리포트 생성
# ==========================================================
print("\n" + "="*60)
print("분석 리포트 생성 중...")
print("="*60)

report = f"""
================================================================================
                드라마 흥행/비흥행 토픽 클러스터 분석 리포트
================================================================================

■ 분석 개요
  - 흥행작: {len(df_hit)}개 드라마
  - 비흥행작: {len(df_flop)}개 드라마
  - 클러스터링 방법: Agglomerative Clustering (cosine, average linkage)

================================================================================
■ 흥행작 토픽 클러스터 분석 (n_groups=4)
================================================================================
"""

for cluster_id in sorted(hit_topic_clusters['클러스터'].unique()):
    cluster_topics = hit_topic_clusters[hit_topic_clusters['클러스터'] == cluster_id]
    topic_nums = cluster_topics['토픽번호'].tolist()
    total_dramas = cluster_topics['드라마수'].sum()
    
    report += f"\n📌 그룹 {cluster_id}: 토픽 {topic_nums} (총 {total_dramas}편)\n"
    report += "-" * 60 + "\n"
    
    for _, row in cluster_topics.iterrows():
        report += f"   토픽 {row['토픽번호']:2d} ({row['드라마수']:3d}편): {row['키워드']}\n"

report += f"""
================================================================================
■ 비흥행작 토픽 클러스터 분석 (n_groups=8)
================================================================================
"""

for cluster_id in sorted(flop_topic_clusters['클러스터'].unique()):
    cluster_topics = flop_topic_clusters[flop_topic_clusters['클러스터'] == cluster_id]
    topic_nums = cluster_topics['토픽번호'].tolist()
    total_dramas = cluster_topics['드라마수'].sum()
    
    report += f"\n📌 그룹 {cluster_id}: 토픽 {topic_nums} (총 {total_dramas}편)\n"
    report += "-" * 60 + "\n"
    
    for _, row in cluster_topics.iterrows():
        report += f"   토픽 {row['토픽번호']:2d} ({row['드라마수']:3d}편): {row['키워드']}\n"

report += f"""
================================================================================
■ 키워드 차집합 분석
================================================================================

[흥행작에만 있는 키워드 (상위 20개)]
"""

for i, (word, score) in enumerate(hit_unique_keywords[:20], 1):
    report += f"  {i:2d}. {word}: {score:.4f}\n"

report += f"""
[비흥행작에만 있는 키워드 (상위 20개)]
"""

for i, (word, score) in enumerate(flop_unique_keywords[:20], 1):
    report += f"  {i:2d}. {word}: {score:.4f}\n"

report += f"""
================================================================================
■ 출력 파일 목록
================================================================================

[분석 결과 CSV]
  - topic_analysis_summary.csv      : 토픽별 분석 요약
  - hit_dramas_with_topics.csv      : 흥행작 드라마별 토픽 할당
  - flop_dramas_with_topics.csv     : 비흥행작 드라마별 토픽 할당
  - hit_topic_clusters.csv          : 흥행작 토픽 클러스터 결과
  - flop_topic_clusters.csv         : 비흥행작 토픽 클러스터 결과
  - hit_cluster_summary.csv         : 흥행작 클러스터 요약
  - flop_cluster_summary.csv        : 비흥행작 클러스터 요약
  - keyword_comparison.csv          : 흥행/비흥행 키워드 차집합

================================================================================
"""

with open(f"{OUTPUT_DIR}/cluster_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  ✓ {OUTPUT_DIR}/cluster_analysis_report.txt")

# ==========================================================
# 완료
# ==========================================================
print("\n" + "="*60)
print(f"분석 완료! 모든 결과가 '{OUTPUT_DIR}/' 폴더에 저장되었습니다.")
print("="*60)

# 저장된 파일 목록 출력
print(f"\n저장된 파일 목록:")
for item in sorted(os.listdir(OUTPUT_DIR)):
    item_path = os.path.join(OUTPUT_DIR, item)
    size = os.path.getsize(item_path)
    if size > 1024*1024:
        print(f"  📄 {item} ({size/1024/1024:.1f} MB)")
    elif size > 1024:
        print(f"  📄 {item} ({size/1024:.1f} KB)")
    else:
        print(f"  📄 {item} ({size} bytes)")
