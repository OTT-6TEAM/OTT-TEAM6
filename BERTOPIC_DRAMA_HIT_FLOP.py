"""
드라마 흥행/비흥행 BERTopic 분석
- 사전 계산된 임베딩(Qwen/Qwen3-Embedding-0.6B) 활용
- hit_score 기준 상위 20% = 흥행, 하위 40% = 비흥행
- 모든 출력물은 '드라마데이터BERTOPIC' 폴더에 저장
"""

import pandas as pd
import numpy as np
import os
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 0. 출력 폴더 생성
# ==========================================================
OUTPUT_DIR = "files/드라마데이터BERTOPIC"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"출력 폴더 생성: {OUTPUT_DIR}/")

# ==========================================================
# 1. 불용어 설정
# ==========================================================
base_stopwords = list(ENGLISH_STOP_WORDS)

additional_drama_stopwords = [
    # ========== 드라마 포맷/메타 ==========
    'tv', 'television', 'show', 'series', 'episode', 'episodes',
    'season', 'seasons', 'installment',
    'pilot', 'finale',

    # ========== 제작/형식 정보 ==========
    'drama', 'dramas',
    'network', 'broadcast', 'air', 'airs',
    'production', 'produced',
    'creator', 'creators',
    'cast', 'crew',
    'actor', 'actors', 'actress', 'actresses',
    'director', 'directors',
    'writer', 'writers',

    # ========== 줄거리 서술 상투어 ==========
    'story', 'stories', 'plot',
    'follows', 'following',
    'centers', 'centred', 'revolves',
    'tells', 'depicts', 'chronicles',
    'focuses', 'explores',
    'takes', 'place',
    'begins', 'starts', 'ends',
    'finds', 'discovers', 'faces', 'way', 'actually', 'la',

    # ========== 일반적 시간 표현 ==========
    'time', 'times',
    'day', 'days',
    'year', 'years',
    'night', 'nights',
    'past', 'present', 'future',
    'later', 'earlier', 'soon',

    # ========== 순서/전개 표현 ==========
    'first', 'second', 'third',
    'last', 'next', 'previous',
    'early', 'late',

    # ========== 일반적 인물 지칭 ==========
    'man', 'woman', 'men', 'women',
    'person', 'people',
    'group', 'groups',
    'team', 'teams',
    'members', 'characters',
    # ========== 너무 일반적인 사건 형용사 ==========
    'high', 'characters', 'just', 'new',
    # ========== 너무 일반적인 사건 동사 ==========
    'life', 'lives',
    'work', 'works',
    'deal', 'deals', 'step', 'gets','decides',
    'struggle', 'struggles', 'make', 'sees', 'set',
    # ========== 고유명사 ==========
    'ryan', 'henry', 'james', 'xun', 'gu', 'ma ri', 'ri', 'ma', 'fernanda', 'rosendo', 'tyler',
    'carmina','mariela', 'lou'
    # 불용어에 추가 가능
    'öykü', 'demir', 'hanzawa', 'leonardo', 'damián', 'eva', 'elisa', 'esteban', 'tori', "eliseo", "sam", "ellen","charlotte", "jarndyce","alex", 
]

english_stopwords_drama = list(set(base_stopwords + additional_drama_stopwords))
print(f"추가 불용어 수: {len(additional_drama_stopwords)}개")
print(f"최종 불용어 수: {len(english_stopwords_drama)}개")

# ==========================================================
# 2. 데이터 로드 및 전처리
# ==========================================================
print("\n" + "="*60)
print("데이터 로드 중...")
print("="*60)

# 데이터 로드 (경로는 실제 환경에 맞게 수정 필요)
drama_df = pd.read_parquet(r"files/final_files/drama/drama_text_embedding_qwen3.parquet")
hit_score_df = pd.read_parquet("files/final_files/00_hit_score.parquet")

print(f"드라마 데이터: {len(drama_df)}개")
print(f"Hit Score 데이터: {len(hit_score_df)}개")

# Left Join
df_merged = drama_df.merge(hit_score_df, on='imdb_id', how='left')
print(f"병합 후 데이터: {len(df_merged)}개")

# hit_score가 있는 데이터만 필터링
df_with_score = df_merged[df_merged['hit_score'].notna()].copy()
print(f"hit_score가 있는 데이터: {len(df_with_score)}개")

# ==========================================================
# 3. 흥행/비흥행 분류 (상위 20%, 하위 20%)
# ==========================================================
print("\n" + "="*60)
print("흥행/비흥행 분류 중...")
print("="*60)

# 퍼센타일 계산
hit_threshold = df_with_score['hit_score'].quantile(0.80)  # 상위 20% 경계
flop_threshold = df_with_score['hit_score'].quantile(0.40)  # 하위 20% 경계

print(f"상위 20% 경계 (hit_score >= {hit_threshold:.4f}): 흥행")
print(f"하위 40% 경계 (hit_score <= {flop_threshold:.4f}): 비흥행")

# 분류
df_hit = df_with_score[df_with_score['hit_score'] >= hit_threshold].copy()
df_flop = df_with_score[df_with_score['hit_score'] <= flop_threshold].copy()

print(f"\n흥행작 수: {len(df_hit)}개")
print(f"비흥행작 수: {len(df_flop)}개")

# ==========================================================
# 4. 임베딩 추출
# ==========================================================
print("\n" + "="*60)
print("임베딩 추출 중...")
print("="*60)

# embedding 컬럼에서 numpy array로 변환
embeddings_hit = np.vstack(df_hit['embedding'].values)
embeddings_flop = np.vstack(df_flop['embedding'].values)

print(f"흥행작 임베딩 shape: {embeddings_hit.shape}")
print(f"비흥행작 임베딩 shape: {embeddings_flop.shape}")

# 텍스트 준비 (combined_text 사용)
texts_hit = df_hit['combined_text'].tolist()
texts_flop = df_flop['combined_text'].tolist()

# ==========================================================
# 5. BERTopic 모델 생성 함수
# ==========================================================

# 임베딩 모델 (BERTopic 내부용 - 실제로는 사전 계산된 임베딩 사용)
embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

def create_bertopic_model(n_neighbors, min_cluster_size, stopwords):
    """
    BERTopic 모델 생성
    
    Args:
        n_neighbors: UMAP 이웃 수 (작을수록 세밀한 토픽)
        min_cluster_size: HDBSCAN 최소 클러스터 크기 (작을수록 토픽 수 증가)
        stopwords: 불용어 리스트
    """
    # CountVectorizer 설정
    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    return BERTopic(
        embedding_model=embedding_model,
        
        # UMAP: 차원 축소
        umap_model=UMAP(
            n_neighbors=n_neighbors,
            n_components=10,
            min_dist=0.05,
            metric='cosine',
            random_state=42
        ),
        
        # HDBSCAN: 밀도 기반 클러스터링
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='leaf',
            prediction_data=True
        ),
        
        vectorizer_model=vectorizer_model,
        verbose=True
    )

# ==========================================================
# 6. 흥행작 토픽 분석
# ==========================================================
print("\n" + "="*60)
print("흥행작 BERTopic 분석")
print("="*60)

# 파라미터 설정
hit_n_neighbors = min(10, len(df_hit) - 1)
hit_min_cluster = max(15, len(df_hit) // 100)

print(f"흥행작 수: {len(df_hit)}")
print(f"UMAP n_neighbors: {hit_n_neighbors}")
print(f"HDBSCAN min_cluster_size: {hit_min_cluster}")

# 모델 생성 및 학습
hit_topic_model = create_bertopic_model(hit_n_neighbors, hit_min_cluster, english_stopwords_drama)

print("\n흥행작 BERTopic 모델 학습 중...")
# BERTopic fit_transform 시 texts를 줄거리만으로 변경
texts_hit_for_ctfidf = df_hit['overview'].tolist()  # 줄거리만

topics_hit, probs_hit = hit_topic_model.fit_transform(
    texts_hit_for_ctfidf,  # ← c-TF-IDF용 텍스트 (줄거리만)
    embeddings=embeddings_hit  # ← 임베딩은 기존 것 사용 (장르+줄거리)
)

# documents 에는 텍스트를, embeddings 에는 벡터를 넣습니다.
new_topics_hit = hit_topic_model.reduce_outliers(
    documents=texts_hit_for_ctfidf,           # 첫 번째 인자: 반드시 텍스트 리스트
    topics=topics_hit,            # 두 번째 인자: 기존 토픽 결과
    strategy="embeddings",          # 전략 선택
    embeddings=embeddings_hit,    # 임베딩 벡터 직접 전달 (속도 향상)
    threshold=0.6                   # 유사도 문턱값
)

# 3. 결과 반영 (필수)
hit_topic_model.update_topics(
    texts_hit_for_ctfidf,
    topics=new_topics_hit,
    vectorizer_model = CountVectorizer(
        stop_words=english_stopwords_drama,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
)

# 결과 출력
hit_topic_info = hit_topic_model.get_topic_info()
print(f"\n[흥행작 토픽 개요] - 총 {len(hit_topic_info) - 1}개 토픽 (Topic -1 제외)")
print(hit_topic_info)

# 각 토픽별 키워드 출력
print("\n[흥행작 토픽별 상위 키워드]")
for topic_id in hit_topic_info['Topic'].values:
    if topic_id != -1:  # 노이즈 토픽 제외
        keywords = hit_topic_model.get_topic(topic_id)
        keyword_str = ", ".join([f"{word}({score:.3f})" for word, score in keywords[:10]])
        print(f"Topic {topic_id}: {keyword_str}")

# ==========================================================
# 7. 비흥행작 토픽 분석
# ==========================================================
print("\n" + "="*60)
print("비흥행작 BERTopic 분석")
print("="*60)

# 파라미터 설정
flop_n_neighbors = min(10, len(df_flop) - 1)
flop_min_cluster = max(15, len(df_flop) // 100)

print(f"비흥행작 수: {len(df_flop)}")
print(f"UMAP n_neighbors: {flop_n_neighbors}")
print(f"HDBSCAN min_cluster_size: {flop_min_cluster}")

# 모델 생성 및 학습
flop_topic_model = create_bertopic_model(flop_n_neighbors, flop_min_cluster, english_stopwords_drama)

print("\n비흥행작 BERTopic 모델 학습 중...")
# ★★★ 수정: 줄거리만 사용 ★★★
texts_flop_for_ctfidf = df_flop['overview'].tolist()  # 줄거리만

topics_flop, probs_flop = flop_topic_model.fit_transform(
    texts_flop_for_ctfidf,  # ← c-TF-IDF용 텍스트 (줄거리만)
    embeddings=embeddings_flop  # ← 임베딩은 기존 것 사용 (장르+줄거리)
)

# documents 에는 텍스트를, embeddings 에는 벡터를 넣습니다.
new_topics_flop = flop_topic_model.reduce_outliers(
    documents=texts_flop_for_ctfidf,           # 첫 번째 인자: 반드시 텍스트 리스트
    topics=topics_flop,            # 두 번째 인자: 기존 토픽 결과
    strategy="embeddings",          # 전략 선택
    embeddings=embeddings_flop,    # 임베딩 벡터 직접 전달 (속도 향상)
    threshold=0.6                   # 유사도 문턱값
)

# 3. 결과 반영 (필수)
flop_topic_model.update_topics(
    texts_flop_for_ctfidf,
    topics=new_topics_flop,
    vectorizer_model = CountVectorizer(
        stop_words=english_stopwords_drama,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
)

# 결과 출력
flop_topic_info = flop_topic_model.get_topic_info()
print(f"\n[비흥행작 토픽 개요] - 총 {len(flop_topic_info) - 1}개 토픽 (Topic -1 제외)")
print(flop_topic_info)

# 각 토픽별 키워드 출력
print("\n[비흥행작 토픽별 상위 키워드]")
for topic_id in flop_topic_info['Topic'].values:
    if topic_id != -1:  # 노이즈 토픽 제외
        keywords = flop_topic_model.get_topic(topic_id)
        keyword_str = ", ".join([f"{word}({score:.3f})" for word, score in keywords[:10]])
        print(f"Topic {topic_id}: {keyword_str}")

# ==========================================================
# 8. 결과 요약 및 비교
# ==========================================================
print("\n" + "="*60)
print("흥행 vs 비흥행 토픽 비교 요약")
print("="*60)

print(f"\n[흥행작]")
print(f"  - 총 드라마 수: {len(df_hit)}")
print(f"  - 발견된 토픽 수: {len(hit_topic_info) - 1}")
print(f"  - 노이즈(Topic -1) 문서 수: {sum(1 for t in new_topics_hit if t == -1)}")

print(f"\n[비흥행작]")
print(f"  - 총 드라마 수: {len(df_flop)}")
print(f"  - 발견된 토픽 수: {len(flop_topic_info) - 1}")
print(f"  - 노이즈(Topic -1) 문서 수: {sum(1 for t in new_topics_flop if t == -1)}")

# ==========================================================
# 9. 시각화 저장
# ==========================================================
print("\n" + "="*60)
print("시각화 생성 중...")
print("="*60)

# ----- 흥행작 시각화 -----

# 1) 토픽별 키워드 바차트
try:
    fig_hit_barchart = hit_topic_model.visualize_barchart(top_n_topics=10)
    fig_hit_barchart.write_html(f"{OUTPUT_DIR}/hit_topics_barchart.html")
    print(f"  ✓ {OUTPUT_DIR}/hit_topics_barchart.html")
except Exception as e:
    print(f"  ✗ 흥행작 바차트 저장 실패: {e}")

# 2) 토픽 간 거리맵 (Intertopic Distance Map)
try:
    fig_hit_intertopic = hit_topic_model.visualize_topics()
    fig_hit_intertopic.write_html(f"{OUTPUT_DIR}/hit_topics_intertopic.html")
    print(f"  ✓ {OUTPUT_DIR}/hit_topics_intertopic.html")
except Exception as e:
    print(f"  ✗ 흥행작 거리맵 저장 실패: {e}")

# 3) 계층적 토픽 구조 (Hierarchical Topics)
try:
    hierarchical_topics_hit = hit_topic_model.hierarchical_topics(texts_hit_for_ctfidf)  # ← 수정
    fig_hit_hierarchy = hit_topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics_hit)
    fig_hit_hierarchy.write_html(f"{OUTPUT_DIR}/hit_topics_hierarchy.html")
    print(f"  ✓ {OUTPUT_DIR}/hit_topics_hierarchy.html")
except Exception as e:
    print(f"  ✗ 흥행작 계층구조 저장 실패: {e}")

# 4) 토픽 히트맵 (Topic Similarity Heatmap)
try:
    fig_hit_heatmap = hit_topic_model.visualize_heatmap()
    fig_hit_heatmap.write_html(f"{OUTPUT_DIR}/hit_topics_heatmap.html")
    print(f"  ✓ {OUTPUT_DIR}/hit_topics_heatmap.html")
except Exception as e:
    print(f"  ✗ 흥행작 히트맵 저장 실패: {e}")

# 5) 문서-토픽 분포 (Document Distribution)
try:
    fig_hit_docs = hit_topic_model.visualize_documents(
        texts_hit_for_ctfidf,  # ← 수정
        embeddings=embeddings_hit,
        hide_annotations=True
    )
    fig_hit_docs.write_html(f"{OUTPUT_DIR}/hit_topics_documents.html")
    print(f"  ✓ {OUTPUT_DIR}/hit_topics_documents.html")
except Exception as e:
    print(f"  ✗ 흥행작 문서분포 저장 실패: {e}")

# ----- 비흥행작 시각화 -----

# 1) 토픽별 키워드 바차트
try:
    fig_flop_barchart = flop_topic_model.visualize_barchart(top_n_topics=10)
    fig_flop_barchart.write_html(f"{OUTPUT_DIR}/flop_topics_barchart.html")
    print(f"  ✓ {OUTPUT_DIR}/flop_topics_barchart.html")
except Exception as e:
    print(f"  ✗ 비흥행작 바차트 저장 실패: {e}")

# 2) 토픽 간 거리맵
try:
    fig_flop_intertopic = flop_topic_model.visualize_topics()
    fig_flop_intertopic.write_html(f"{OUTPUT_DIR}/flop_topics_intertopic.html")
    print(f"  ✓ {OUTPUT_DIR}/flop_topics_intertopic.html")
except Exception as e:
    print(f"  ✗ 비흥행작 거리맵 저장 실패: {e}")

# 3) 계층적 토픽 구조
try:
    hierarchical_topics_flop = flop_topic_model.hierarchical_topics(texts_flop_for_ctfidf)  # ← 수정
    fig_flop_hierarchy = flop_topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics_flop)
    fig_flop_hierarchy.write_html(f"{OUTPUT_DIR}/flop_topics_hierarchy.html")
    print(f"  ✓ {OUTPUT_DIR}/flop_topics_hierarchy.html")
except Exception as e:
    print(f"  ✗ 비흥행작 계층구조 저장 실패: {e}")

# 4) 토픽 히트맵
try:
    fig_flop_heatmap = flop_topic_model.visualize_heatmap()
    fig_flop_heatmap.write_html(f"{OUTPUT_DIR}/flop_topics_heatmap.html")
    print(f"  ✓ {OUTPUT_DIR}/flop_topics_heatmap.html")
except Exception as e:
    print(f"  ✗ 비흥행작 히트맵 저장 실패: {e}")

# 5) 문서-토픽 분포
try:
    fig_flop_docs = flop_topic_model.visualize_documents(
        texts_flop_for_ctfidf,  # ← 수정
        embeddings=embeddings_flop,
        hide_annotations=True
    )
    fig_flop_docs.write_html(f"{OUTPUT_DIR}/flop_topics_documents.html")
    print(f"  ✓ {OUTPUT_DIR}/flop_topics_documents.html")
except Exception as e:
    print(f"  ✗ 비흥행작 문서분포 저장 실패: {e}")


# ==========================================================
# 10-1. Representative_Docs에 드라마 제목 매핑 (추가 코드)
# ==========================================================
print("\n" + "="*60)
print("Representative_Docs에 드라마 제목 매핑 중...")
print("="*60)

import ast

def map_representative_docs_to_titles(topic_info_df, texts_list, titles_list):
    """
    Representative_Docs의 줄거리를 드라마 제목과 매핑
    
    Args:
        topic_info_df: BERTopic의 get_topic_info() 결과 DataFrame
        texts_list: fit_transform에 사용된 텍스트 리스트 (overview)
        titles_list: 대응되는 제목 리스트
    
    Returns:
        제목이 추가된 DataFrame
    """
    # 텍스트 -> 제목 매핑 딕셔너리 생성
    text_to_title = {text: title for text, title in zip(texts_list, titles_list)}
    
    # Representative_Docs_Titles 컬럼 생성
    representative_titles = []
    
    for idx, row in topic_info_df.iterrows():
        if row['Topic'] == -1:
            representative_titles.append([])
            continue
            
        rep_docs = row['Representative_Docs']
        
        # rep_docs가 문자열인 경우 리스트로 변환
        if isinstance(rep_docs, str):
            try:
                rep_docs = ast.literal_eval(rep_docs)
            except:
                rep_docs = [rep_docs]
        
        # 각 대표 문서에 대응하는 제목 찾기
        titles = []
        for doc in rep_docs:
            title = text_to_title.get(doc, "제목 없음")
            titles.append(title)
        
        representative_titles.append(titles)
    
    # 새 컬럼 추가
    topic_info_df = topic_info_df.copy()
    topic_info_df['Representative_Docs_Titles'] = representative_titles
    
    return topic_info_df

# ----- 흥행작 처리 -----
titles_hit = df_hit['title'].tolist()

hit_topic_info_with_titles = map_representative_docs_to_titles(
    hit_topic_info, 
    texts_hit_for_ctfidf, 
    titles_hit
)

# CSV 저장 (리스트를 문자열로 변환)
hit_topic_info_with_titles_csv = hit_topic_info_with_titles.copy()
hit_topic_info_with_titles_csv['Representative_Docs_Titles'] = hit_topic_info_with_titles_csv['Representative_Docs_Titles'].apply(
    lambda x: ' | '.join(x) if isinstance(x, list) else x
)
hit_topic_info_with_titles_csv.to_csv(f"{OUTPUT_DIR}/hit_topic_info_with_titles.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/hit_topic_info_with_titles.csv")

# ----- 비흥행작 처리 -----
titles_flop = df_flop['title'].tolist()

flop_topic_info_with_titles = map_representative_docs_to_titles(
    flop_topic_info, 
    texts_flop_for_ctfidf, 
    titles_flop
)

# CSV 저장
flop_topic_info_with_titles_csv = flop_topic_info_with_titles.copy()
flop_topic_info_with_titles_csv['Representative_Docs_Titles'] = flop_topic_info_with_titles_csv['Representative_Docs_Titles'].apply(
    lambda x: ' | '.join(x) if isinstance(x, list) else x
)
flop_topic_info_with_titles_csv.to_csv(f"{OUTPUT_DIR}/flop_topic_info_with_titles.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/flop_topic_info_with_titles.csv")

# ----- 결과 미리보기 -----
print("\n[흥행작 토픽별 대표 드라마]")
for _, row in hit_topic_info_with_titles.iterrows():
    if row['Topic'] != -1:
        titles_str = ", ".join(row['Representative_Docs_Titles'][:3])
        print(f"  Topic {row['Topic']}: {titles_str}")

print("\n[비흥행작 토픽별 대표 드라마]")
for _, row in flop_topic_info_with_titles.iterrows():
    if row['Topic'] != -1:
        titles_str = ", ".join(row['Representative_Docs_Titles'][:3])
        print(f"  Topic {row['Topic']}: {titles_str}")


# ==========================================================
# 10. 데이터 파일 저장
# ==========================================================
print("\n" + "="*60)
print("데이터 파일 저장 중...")
print("="*60)

# ----- 흥행작 결과 저장 -----

# 1) 드라마별 토픽 할당 결과
df_hit_result = df_hit[['imdb_id', 'title', 'combined_text', 'hit_score']].copy()
df_hit_result['topic'] = new_topics_hit

# ★ 수정: probs 형태에 따라 처리 (에러 방지)
if isinstance(probs_hit, np.ndarray) and probs_hit.ndim == 1:
    df_hit_result['topic_prob'] = probs_hit
else:
    df_hit_result['topic_prob'] = [max(p) if hasattr(p, '__len__') and len(p) > 0 else float(p) for p in probs_hit]

df_hit_result.to_parquet(f"{OUTPUT_DIR}/hit_drama_topics.parquet", index=False)
df_hit_result.to_csv(f"{OUTPUT_DIR}/hit_drama_topics.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/hit_drama_topics.parquet")
print(f"  ✓ {OUTPUT_DIR}/hit_drama_topics.csv")

# 2) 토픽 정보 요약
hit_topic_info.to_csv(f"{OUTPUT_DIR}/hit_topic_info.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/hit_topic_info.csv")

# 3) 토픽별 상세 키워드
hit_keywords_data = []
for topic_id in hit_topic_info['Topic'].values:
    if topic_id != -1:
        keywords = hit_topic_model.get_topic(topic_id)
        for rank, (word, score) in enumerate(keywords[:20], 1):
            hit_keywords_data.append({
                'topic': topic_id,
                'rank': rank,
                'keyword': word,
                'score': score
            })
df_hit_keywords = pd.DataFrame(hit_keywords_data)
df_hit_keywords.to_csv(f"{OUTPUT_DIR}/hit_topic_keywords.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/hit_topic_keywords.csv")

# ----- 비흥행작 결과 저장 -----

# 1) 드라마별 토픽 할당 결과
df_flop_result = df_flop[['imdb_id', 'title', 'combined_text', 'hit_score']].copy()
df_flop_result['topic'] = new_topics_flop

# ★ 수정: probs 형태에 따라 처리 (에러 방지)
if isinstance(probs_flop, np.ndarray) and probs_flop.ndim == 1:
    df_flop_result['topic_prob'] = probs_flop
else:
    df_flop_result['topic_prob'] = [max(p) if hasattr(p, '__len__') and len(p) > 0 else float(p) for p in probs_flop]

df_flop_result.to_parquet(f"{OUTPUT_DIR}/flop_drama_topics.parquet", index=False)
df_flop_result.to_csv(f"{OUTPUT_DIR}/flop_drama_topics.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/flop_drama_topics.parquet")
print(f"  ✓ {OUTPUT_DIR}/flop_drama_topics.csv")

# 2) 토픽 정보 요약
flop_topic_info.to_csv(f"{OUTPUT_DIR}/flop_topic_info.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/flop_topic_info.csv")

# 3) 토픽별 상세 키워드
flop_keywords_data = []
for topic_id in flop_topic_info['Topic'].values:
    if topic_id != -1:
        keywords = flop_topic_model.get_topic(topic_id)
        for rank, (word, score) in enumerate(keywords[:20], 1):
            flop_keywords_data.append({
                'topic': topic_id,
                'rank': rank,
                'keyword': word,
                'score': score
            })
df_flop_keywords = pd.DataFrame(flop_keywords_data)
df_flop_keywords.to_csv(f"{OUTPUT_DIR}/flop_topic_keywords.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ {OUTPUT_DIR}/flop_topic_keywords.csv")

# ==========================================================
# 11. BERTopic 모델 저장 (선택적 - 디스크 공간 확인 필요)
# ==========================================================
print("\n" + "="*60)
print("BERTopic 모델 저장 중...")
print("="*60)

# ★ 수정: safetensors 사용 + embedding_model 제외 (용량 절약)
try:
    hit_topic_model.save(
        f"{OUTPUT_DIR}/hit_bertopic_model", 
        serialization="safetensors", 
        save_ctfidf=True, 
        save_embedding_model=False
    )
    print(f"  ✓ {OUTPUT_DIR}/hit_bertopic_model/")
except Exception as e:
    print(f"  ✗ 흥행작 모델 저장 실패: {e}")

try:
    flop_topic_model.save(
        f"{OUTPUT_DIR}/flop_bertopic_model", 
        serialization="safetensors", 
        save_ctfidf=True, 
        save_embedding_model=False
    )
    print(f"  ✓ {OUTPUT_DIR}/flop_bertopic_model/")
except Exception as e:
    print(f"  ✗ 비흥행작 모델 저장 실패: {e}")

# ==========================================================
# 12. 분석 요약 리포트 저장
# ==========================================================
print("\n" + "="*60)
print("분석 요약 리포트 생성 중...")
print("="*60)

report = f"""
================================================================================
                    드라마 흥행/비흥행 BERTopic 분석 리포트
================================================================================

■ 분석 개요
  - 분석 대상: hit_score가 있는 드라마 {len(df_with_score)}개
  - 흥행 기준: hit_score 상위 20% (>= {hit_threshold:.4f})
  - 비흥행 기준: hit_score 하위 40% (<= {flop_threshold:.4f})
  - 임베딩 모델: Qwen/Qwen3-Embedding-0.6B

================================================================================
■ 흥행작 분석 결과
================================================================================
  - 분석 대상 수: {len(df_hit)}개
  - 발견된 토픽 수: {len(hit_topic_info) - 1}개
  - 노이즈(미분류) 문서 수: {sum(1 for t in new_topics_hit if t == -1)}개
  - 클러스터링 파라미터:
    · UMAP n_neighbors: {hit_n_neighbors}
    · HDBSCAN min_cluster_size: {hit_min_cluster}

  [토픽별 요약]
"""

for _, row in hit_topic_info.iterrows():
    if row['Topic'] != -1:
        keywords = hit_topic_model.get_topic(row['Topic'])
        top_keywords = ", ".join([w for w, s in keywords[:5]])
        report += f"    Topic {row['Topic']}: {row['Count']}개 문서 - {top_keywords}\n"

report += f"""
================================================================================
■ 비흥행작 분석 결과
================================================================================
  - 분석 대상 수: {len(df_flop)}개
  - 발견된 토픽 수: {len(flop_topic_info) - 1}개
  - 노이즈(미분류) 문서 수: {sum(1 for t in new_topics_flop if t == -1)}개
  - 클러스터링 파라미터:
    · UMAP n_neighbors: {flop_n_neighbors}
    · HDBSCAN min_cluster_size: {flop_min_cluster}

  [토픽별 요약]
"""

for _, row in flop_topic_info.iterrows():
    if row['Topic'] != -1:
        keywords = flop_topic_model.get_topic(row['Topic'])
        top_keywords = ", ".join([w for w, s in keywords[:5]])
        report += f"    Topic {row['Topic']}: {row['Count']}개 문서 - {top_keywords}\n"

report += f"""
================================================================================
■ 출력 파일 목록
================================================================================

[시각화 파일 - HTML (브라우저에서 열기)]
  흥행작:
    - hit_topics_barchart.html    : 토픽별 상위 키워드 막대그래프
    - hit_topics_intertopic.html  : 토픽 간 거리/유사도 맵
    - hit_topics_hierarchy.html   : 토픽 계층 구조 (덴드로그램)
    - hit_topics_heatmap.html     : 토픽 간 유사도 히트맵
    - hit_topics_documents.html   : 문서 분포 시각화 (2D 산점도)

  비흥행작:
    - flop_topics_barchart.html   : 토픽별 상위 키워드 막대그래프
    - flop_topics_intertopic.html : 토픽 간 거리/유사도 맵
    - flop_topics_hierarchy.html  : 토픽 계층 구조 (덴드로그램)
    - flop_topics_heatmap.html    : 토픽 간 유사도 히트맵
    - flop_topics_documents.html  : 문서 분포 시각화 (2D 산점도)

[데이터 파일 - CSV/Parquet]
  흥행작:
    - hit_drama_topics.csv/parquet : 각 드라마의 토픽 할당 결과
    - hit_topic_info.csv           : 토픽 요약 정보 (문서 수, 대표 키워드)
    - hit_topic_keywords.csv       : 토픽별 상위 20개 키워드 및 점수

  비흥행작:
    - flop_drama_topics.csv/parquet : 각 드라마의 토픽 할당 결과
    - flop_topic_info.csv           : 토픽 요약 정보 (문서 수, 대표 키워드)
    - flop_topic_keywords.csv       : 토픽별 상위 20개 키워드 및 점수

[모델 파일 - 재사용 가능]
    - hit_bertopic_model/  : 흥행작 BERTopic 모델
    - flop_bertopic_model/ : 비흥행작 BERTopic 모델

================================================================================
■ 파일 설명
================================================================================

1. *_topics_barchart.html
   - 각 토픽의 대표 키워드와 c-TF-IDF 점수를 막대그래프로 표시
   - 토픽의 주제를 빠르게 파악할 때 사용

2. *_topics_intertopic.html
   - 토픽들을 2D 공간에 배치하여 유사한 토픽끼리 가까이 위치
   - 원의 크기는 해당 토픽의 문서 수를 나타냄
   - 토픽 간 관계를 파악할 때 사용

3. *_topics_hierarchy.html
   - 토픽들의 계층적 클러스터링 결과 (덴드로그램)
   - 유사한 토픽들이 어떻게 그룹화되는지 확인

4. *_topics_heatmap.html
   - 토픽 간 코사인 유사도를 히트맵으로 표시
   - 어떤 토픽들이 서로 유사한지 정량적으로 파악

5. *_topics_documents.html
   - 모든 문서를 2D 공간에 시각화 (UMAP 차원 축소)
   - 각 점은 하나의 드라마, 색상은 할당된 토픽
   - 클러스터링이 잘 되었는지 시각적으로 확인

6. *_drama_topics.csv
   - imdb_id, title: 드라마 식별 정보
   - combined_text: 분석에 사용된 텍스트 (줄거리+장르)
   - hit_score: 흥행 점수
   - topic: 할당된 토픽 번호 (-1은 노이즈/미분류)
   - topic_prob: 해당 토픽에 속할 확률

7. *_topic_info.csv
   - Topic: 토픽 번호
   - Count: 해당 토픽에 속한 문서 수
   - Name: 토픽 대표 키워드 조합

8. *_topic_keywords.csv
   - topic: 토픽 번호
   - rank: 키워드 순위 (1~20)
   - keyword: 키워드
   - score: c-TF-IDF 점수 (높을수록 해당 토픽에서 중요)

================================================================================
"""

with open(f"{OUTPUT_DIR}/analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  ✓ {OUTPUT_DIR}/analysis_report.txt")

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
    if os.path.isdir(item_path):
        print(f"  📁 {item}/")
    else:
        size = os.path.getsize(item_path)
        if size > 1024*1024:
            print(f"{item} ({size/1024/1024:.1f} MB)")
        elif size > 1024:
            print(f"{item} ({size/1024:.1f} KB)")
        else:
            print(f"{item} ({size} bytes)")