# =============================================================================
# 필요한 라이브러리 import
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

# =============================================================================
# 불용어 리스트 정의
# =============================================================================

# ==========================================================
# 1. 기본 불용어
# ==========================================================
# sklearn 내장 불용어 (318개)
base_stopwords = list(ENGLISH_STOP_WORDS)
print(f"sklearn 내장 불용어 수: {len(base_stopwords)}개")

# ==========================================================
# 2. 영화 도메인 특화 불용어
# ==========================================================
# 추가 불용어: 영화 도메인 특화
additional_movie_stopwords = [
    # ========== 영화 도메인 공통어 ==========
    'film', 'films', 'movie', 'movies', 'story', 'stories',
    'character', 'characters', 'scene', 'scenes', 'plot',
    'protagonist', 'audience', 'viewer', 'viewers',
    'series', 'sequel', 'part', 'chapter',
    'director', 'actor', 'actress', 'cast', 'crew',
    'documentary', 'footage', 'screen',
    
    # ========== 줄거리 서술 상투어 ==========
    'based', 'true', 'real', 'events', 'set',
    'follows', 'following', 'centers', 'revolves', 'tells',
    'takes', 'place', 'turns', 'finds', 'discovers',
    'begins', 'starts', 'ends', 'leads', 'brings',
    
    # ========== 일반적 시간/수량 표현 ==========
    'time', 'times', 'year', 'years', 'day', 'days', 'night', 'nights',
    'moment', 'moments', 'later', 'ago', 'soon',
    'one', 'two', 'three', 'first', 'second', 'third', 'last',
    
    # ========== 일반적 인물 지칭 ==========
    'man', 'woman', 'men', 'women', 'people', 'person', 'guy',
    'group', 'team', 'crew', 'members',
]

# 최종 불용어 리스트
english_stopwords_movie = list(set(base_stopwords + additional_movie_stopwords))
print(f"추가 불용어 수: {len(additional_movie_stopwords)}개")
print(f"최종 불용어 수: {len(english_stopwords_movie)}개")

# =============================================================================
# Pandas 출력 옵션 설정
# =============================================================================
pd.set_option('display.max_columns', None)      # 모든 컬럼 표시
pd.set_option('display.max_colwidth', None)     # 컬럼 내용 전체 표시 (잘림 방지)
pd.set_option('display.width', None)            # 출력 너비 제한 해제
pd.set_option('display.max_rows', None)         # 모든 행 표시

# =============================================================================
# 1. 데이터 로드 (사전 계산된 임베딩 포함)
# =============================================================================
print("\n" + "=" * 60)
print("1단계: 데이터 로드 (사전 계산된 임베딩 포함)")
print("=" * 60)

# 임베딩이 포함된 영화 데이터 로드
df = pd.read_parquet("최종데이터셋_영화\movie_with_embeddings_final.parquet")
print(f"df shape: {df.shape}")
print(f"df columns: {df.columns.tolist()}")

# 임베딩 컬럼 확인
print(f"\n임베딩 컬럼 타입: {type(df['embedding'].iloc[0])}")
print(f"임베딩 차원: {len(df['embedding'].iloc[0])}")

# =============================================================================
# 2. 임베딩 데이터 변환
# =============================================================================
print("\n" + "=" * 60)
print("2단계: 임베딩 데이터 변환")
print("=" * 60)

# embedding 컬럼을 numpy array로 변환
# parquet에서 로드 시 리스트 형태일 수 있으므로 변환 필요
all_embeddings = np.array(df['embedding'].tolist())
print(f"임베딩 shape: {all_embeddings.shape}")

# =============================================================================
# 3. text 컬럼 확인/생성
# =============================================================================
print("\n" + "=" * 60)
print("3단계: text 컬럼 확인")
print("=" * 60)

# text 컬럼이 있는지 확인, 없으면 생성
if 'text' not in df.columns:
    print("text 컬럼이 없습니다. 생성 중...")
    df['overview'] = df['overview'].fillna('')
    if 'genres_combined' in df.columns:
        df['genres_combined'] = df['genres_combined'].fillna('')
        df['text'] = "Genres: " + df['genres_combined'] + ". Overview: " + df['overview']
    else:
        df['text'] = df['overview']
else:
    print("text 컬럼이 이미 존재합니다.")

# 빈 텍스트 제거
df = df[df['text'].str.strip() != ''].reset_index(drop=True)
all_embeddings = np.array(df['embedding'].tolist())  # 필터링 후 임베딩도 다시 추출

print(f"유효한 텍스트가 있는 영화 수: {len(df)}")
print(f"최종 임베딩 shape: {all_embeddings.shape}")

print("\ntext 컬럼 예시:")
for i in range(min(3, len(df))):
    print(f"\n[{i}] {df['text'].iloc[i][:200]}...")

# =============================================================================
# 4. CountVectorizer 설정
# =============================================================================
print("\n" + "=" * 60)
print("4단계: CountVectorizer 설정")
print("=" * 60)

# CountVectorizer: BERTopic 내부 c-TF-IDF 계산용
vectorizer_model = CountVectorizer(
    stop_words=english_stopwords_movie,  # 불용어 제거
    ngram_range=(1, 2),                  # 1-gram + 2-gram 추출
    min_df=3,                            # 최소 3개 문서에 등장해야 포함
    max_df=0.85                          # 85% 이상 문서에 등장하면 제외
)

print("CountVectorizer 설정 완료")

# =============================================================================
# 5. BERTopic 모델 생성 함수 정의
# =============================================================================
print("\n" + "=" * 60)
print("5단계: BERTopic 모델 생성 함수 정의")
print("=" * 60)


def create_bertopic_model(n_neighbors, min_cluster_size):
    """
    BERTopic 모델 생성
    
    Args:
        n_neighbors: UMAP 이웃 수 (작을수록 세밀한 토픽)
        min_cluster_size: HDBSCAN 최소 클러스터 크기 (작을수록 토픽 수 증가)
    
    Returns:
        BERTopic 모델 객체
    """
    return BERTopic(
        # 임베딩 모델: 사전 계산된 임베딩을 전달하므로 None으로 설정
        embedding_model=None,
        
        # ========== UMAP: 차원 축소 ==========
        umap_model=UMAP(
            n_neighbors=n_neighbors,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        ),
        
        # ========== HDBSCAN: 밀도 기반 클러스터링 ==========
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
            prediction_data=True
        ),
        
        # c-TF-IDF 키워드 추출용
        vectorizer_model=vectorizer_model,
        
        # 학습 과정 출력
        verbose=True
    )


print("BERTopic 모델 생성 함수 정의 완료")

# =============================================================================
# 6. 전체 데이터 BERTopic 분석
# =============================================================================
print("\n" + "=" * 60)
print("6단계: 전체 BERTopic 분석")
print("=" * 60)

# 파라미터 설정
n_neighbors = min(15, len(df) - 1)
min_cluster_size = max(15, len(df) // 100)

print(f"전체 영화 수: {len(df)}")
print(f"UMAP n_neighbors: {n_neighbors}")
print(f"HDBSCAN min_cluster_size: {min_cluster_size}")

# 모델 생성 및 학습
topic_model = create_bertopic_model(n_neighbors, min_cluster_size)

print("\n전체 BERTopic 모델 학습 중...")
topics, probs = topic_model.fit_transform(
    df['text'].tolist(),
    embeddings=all_embeddings  # 사전 계산된 임베딩 사용
)

# 결과 저장
df['topic'] = topics

# =============================================================================
# 7. 결과 출력
# =============================================================================
print("\n" + "=" * 60)
print("7단계: 결과 출력")
print("=" * 60)

topic_info = topic_model.get_topic_info()
print(f"\n[전체 토픽 개요] - 총 {len(topic_info) - 1}개 토픽")
print(topic_info)

# 각 토픽별 상세 키워드 출력
print("\n" + "=" * 60)
print("각 토픽별 상세 키워드")
print("=" * 60)

for topic_id in topic_info['Topic'].unique():
    if topic_id != -1:  # -1은 노이즈(outlier)
        topic_words = topic_model.get_topic(topic_id)
        print(f"\n토픽 {topic_id}: {[word for word, _ in topic_words[:10]]}")

# =============================================================================
# 8. 결과 저장
# =============================================================================
print("\n" + "=" * 60)
print("8단계: 결과 저장")
print("=" * 60)

# 토픽이 할당된 데이터프레임 저장 (임베딩 컬럼 제외하여 용량 절약)
df_save = df.drop(columns=['embedding'])
df_save.to_parquet("movie_with_topics.parquet", index=False)
print("movie_with_topics.parquet 저장 완료")

# 토픽 정보 저장
topic_info.to_csv("movie_topic_info.csv", index=False)
print("movie_topic_info.csv 저장 완료")

print("\n" + "=" * 60)
print("분석 완료!")
print("=" * 60)