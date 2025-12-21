# =============================================================================
# 필요한 라이브러리 import
# =============================================================================
import pandas as pd
from sentence_transformers import SentenceTransformer
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
base_stopwords = list(ENGLISH_STOP_WORDS)

# ==========================================================
# 2. TV 드라마 도메인 특화 불용어
# ==========================================================
additional_drama_stopwords = [
    # ========== 드라마 포맷/메타 ==========
    'tv', 'television', 'show', 'series', 'episode', 'episodes',
    'season', 'seasons', 'installment',
    'pilot', 'finale', 'Genres', 'genres',
    
    # ========== 제작/형식 정보 ==========
    'drama', 'dramas',   # ⚠️ 장르 자체가 이미 컬럼에 있으므로 제거
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
    'finds', 'discovers', 'faces',
    
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
    
    # ========== 일반적 인물 지칭 (⚠️ 관계 단어는 제외) ==========
    'man', 'woman', 'men', 'women',
    'person', 'people',
    'group', 'groups',
    'team', 'teams',
    'members',
    
    # ========== 너무 일반적인 사건 동사 ==========
    'life', 'lives',
    'work', 'works',
    'deal', 'deals',
    'struggle', 'struggles',
]

english_stopwords_drama = list(set(base_stopwords + additional_drama_stopwords))
print(f"기본 불용어 수: {len(base_stopwords)}개")
print(f"추가 불용어 수: {len(additional_drama_stopwords)}개")
print(f"최종 불용어 수: {len(english_stopwords_drama)}개")

# =============================================================================
# Pandas 출력 옵션 설정
# =============================================================================
pd.set_option('display.max_columns', None)      # 모든 컬럼 표시
pd.set_option('display.max_colwidth', None)     # 컬럼 내용 전체 표시 (잘림 방지)
pd.set_option('display.width', None)            # 출력 너비 제한 해제
pd.set_option('display.max_rows', None)         # 모든 행 표시

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n" + "=" * 60)
print("1단계: 데이터 로드")
print("=" * 60)

# 드라마 메인 데이터 로드
drama_final = pd.read_parquet("최종데이터셋_드라마/drama_final.parquet")
print(f"drama_final shape: {drama_final.shape}")
print(f"drama_final columns: {drama_final.columns.tolist()}")

# 장르 데이터 로드
drama_genres_final = pd.read_parquet("최종데이터셋_드라마/drama_genres_final.parquet")
print(f"\ndrama_genres_final shape: {drama_genres_final.shape}")
print(f"drama_genres_final columns: {drama_genres_final.columns.tolist()}")

# =============================================================================
# 2. 장르 데이터 병합 (imdb_id 기준으로 장르 그룹화)
# =============================================================================
print("\n" + "=" * 60)
print("2단계: 장르 데이터 병합")
print("=" * 60)

# 같은 imdb_id의 장르들을 하나의 문자열로 합치기
genres_grouped = drama_genres_final.groupby('imdb_id')['genre'].apply(
    lambda x: ', '.join(x.astype(str))
).reset_index()
genres_grouped.columns = ['imdb_id', 'genres_combined']

print(f"장르 그룹화 완료: {len(genres_grouped)}개 드라마")
print("\n장르 결합 예시:")
print(genres_grouped.head())

df = drama_final.merge(genres_grouped, on='imdb_id', how='left')
print(f"\n병합 후 shape: {df.shape}")

# =============================================================================
# 3. text 컬럼 생성 (줄거리 + 장르)
# =============================================================================
print("\n" + "=" * 60)
print("3단계: text 컬럼 생성 (줄거리 + 장르)")
print("=" * 60)

# 결측치 처리
df['overview'] = df['overview'].fillna('')
df['genres_combined'] = df['genres_combined'].fillna('')

# text 컬럼 생성
df['text'] = (
    "Genres: " + df['genres_combined'] + ". " +
    "Overview: " + df['overview']
)

# text가 빈 문자열인 행 제거
df = df[df['text'].str.strip() != ''].reset_index(drop=True)

print(f"유효한 텍스트가 있는 드라마 수: {len(df)}")
print("\ntext 컬럼 예시:")
for i in range(min(3, len(df))):
    print(f"\n[{i}] {df['text'].iloc[i][:200]}...")  # 앞 200자만 출력

# =============================================================================
# 4. 임베딩 모델 로드 및 임베딩 생성
# =============================================================================
print("\n" + "=" * 60)
print("4단계: 임베딩 생성")
print("=" * 60)

print("임베딩 모델 로드 중: BAAI/bge-large-en-v1.5")
print("잠시 기다려주세요...")

embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
print(f"모델 로드 완료! 임베딩 차원: {embedding_model.get_sentence_embedding_dimension()}")


def prepare_embeddings(texts, model):
    """
    주어진 텍스트 리스트를 벡터(임베딩)으로 변환
    
    Args:
        texts: 변환할 텍스트 리스트
        model: SentenceTransformer 모델 객체
    
    Returns:
        numpy array (문서 수, 임베딩 차원)
    """
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )


# 전체 분석 대상 임베딩 생성
print("\n모든 드라마에 대한 임베딩 생성 중...")
all_embeddings = prepare_embeddings(df['text'].tolist(), embedding_model)
print(f"임베딩 shape: {all_embeddings.shape}")

# =============================================================================
# 5. CountVectorizer 설정
# =============================================================================
print("\n" + "=" * 60)
print("5단계: CountVectorizer 설정")
print("=" * 60)

# CountVectorizer: BERTopic 내부 c-TF-IDF 계산용
# - 각 토픽(클러스터)별 대표 키워드를 추출할 때 사용
vectorizer_model = CountVectorizer(
    stop_words=english_stopwords_drama,  # 불용어 제거
    ngram_range=(1, 2),                  # 1-gram + 2-gram 추출
    min_df=3,                            # 최소 3개 문서에 등장해야 포함
    max_df=0.85                          # 85% 이상 문서에 등장하면 제외
)

print("CountVectorizer 설정 완료")

# =============================================================================
# 6. BERTopic 모델 생성 함수 정의
# =============================================================================
print("\n" + "=" * 60)
print("6단계: BERTopic 모델 생성 함수 정의")
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
        # 고차원(1024) → 저차원(5)으로 축소하여 클러스터링 효율화
        umap_model=UMAP(
            # n_neighbors: 각 점 주변의 이웃 수
            # - 작을수록: 지역적 구조 강조 → 세밀한 클러스터
            # - 클수록: 전역적 구조 강조 → 큰 클러스터
            n_neighbors=n_neighbors,
            
            # n_components: 축소할 차원 수
            # - 너무 낮으면 정보 손실, 너무 높으면 클러스터링 어려움
            n_components=5,
            
            # min_dist: 저차원 공간에서 점 사이 최소 거리
            # - 0.0: 밀집된 클러스터 형성 (클러스터링에 적합)
            # - 클수록: 점들이 더 퍼짐
            min_dist=0.0,
            
            # metric: 거리 측정 방식
            # - 'cosine': 텍스트 임베딩에 적합 (방향 유사도)
            metric='cosine',
            
            # random_state: 재현 가능성을 위한 시드
            random_state=42
        ),
        
        # ========== HDBSCAN: 밀도 기반 클러스터링 ==========
        # K-Means와 달리 클러스터 수를 자동으로 결정
        hdbscan_model=HDBSCAN(
            # min_cluster_size: 클러스터로 인정받기 위한 최소 문서 수
            # - 작을수록: 토픽 수 증가, 세분화
            # - 클수록: 토픽 수 감소, 노이즈(outlier) 증가
            min_cluster_size=min_cluster_size,
            
            # min_samples: 코어 포인트가 되기 위한 최소 이웃 수
            # - 밀도 기준 (클수록 더 엄격한 밀도 요구)
            min_samples=5,
            
            # metric: 거리 측정 방식
            # - UMAP 축소 후에는 'euclidean' 사용
            metric='euclidean',
            
            # prediction_data: 새 문서 예측을 위한 데이터 저장
            prediction_data=True
        ),
        
        # c-TF-IDF 키워드 추출용
        vectorizer_model=vectorizer_model,
        
        # 학습 과정 출력
        verbose=True
    )


print("BERTopic 모델 생성 함수 정의 완료")

# =============================================================================
# 7. 전체 데이터 BERTopic 분석
# =============================================================================
print("\n" + "=" * 60)
print("7단계: 전체 BERTopic 분석")
print("=" * 60)

# 파라미터 설정
n_neighbors = min(15, len(df) - 1)
min_cluster_size = max(15, len(df) // 100)

print(f"전체 드라마 수: {len(df)}")
print(f"UMAP n_neighbors: {n_neighbors}")
print(f"HDBSCAN min_cluster_size: {min_cluster_size}")

# 모델 생성 및 학습
topic_model = create_bertopic_model(n_neighbors, min_cluster_size)

print("\n전체 BERTopic 모델 학습 중...")
topics, probs = topic_model.fit_transform(
    df['text'].tolist(),
    embeddings=all_embeddings  # 이미 생성된 임베딩 사용
)

# 결과 저장
df['topic'] = topics

# =============================================================================
# 8. 결과 출력
# =============================================================================
print("\n" + "=" * 60)
print("8단계: 결과 출력")
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
# 9. 결과 저장 (선택사항)
# =============================================================================
print("\n" + "=" * 60)
print("9단계: 결과 저장")
print("=" * 60)

# 토픽이 할당된 데이터프레임 저장
df.to_parquet("drama_with_topics.parquet", index=False)
print("drama_with_topics.parquet 저장 완료")

# 토픽 정보 저장
topic_info.to_csv("drama_topic_info.csv", index=False)
print("drama_topic_info.csv 저장 완료")

print("\n" + "=" * 60)
print("분석 완료!")
print("=" * 60)