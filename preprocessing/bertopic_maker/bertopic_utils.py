from .set_up import *

def cluster_topics(topic_model):
    """
    토픽 간 거리를 계산하고 유사한 토픽끼리 그룹화

    Args:
        topic_model: 학습된 BERTopic 모델
        n_groups: 원하는 그룹 수
        label: 출력 시 표시할 라벨 (흥행작/비흥행작)

    Returns:
        topic_clusters: 토픽별 클러스터 정보 DataFrame
    """

    def suggest_n_clusters(n_topics):
        """
        토픽 수에 따른 경험적 추천
        """
        if n_topics <= 5:
            return 2
        elif n_topics <= 10:
            return 3
        elif n_topics <= 20:
            return int(np.sqrt(n_topics))  # √n
        elif n_topics <= 30:
            return int(n_topics / 4)
        else:
            return int(n_topics / 5)

    # 토픽 임베딩(좌표) 추출
    topic_embeddings = topic_model.topic_embeddings_

    # 토픽 정보 (outlier -1 제외)
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()

    # outlier(-1)는 인덱스 0에 있으므로, 실제 토픽은 인덱스 1부터
    valid_embeddings = topic_embeddings[1:len(valid_topics) + 1]

    n_groups = suggest_n_clusters(len(topic_info))

    # 계층적 클러스터링
    clustering = AgglomerativeClustering(
        n_clusters=n_groups,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(valid_embeddings)

    # 결과 정리
    topic_clusters = pd.DataFrame({
        'topic_num': valid_topics,
        'cluster': cluster_labels,
        'cnt': [topic_info[topic_info['Topic'] == t]['Count'].values[0] for t in valid_topics],
        'keyword': [', '.join([w for w, s in topic_model.get_topic(t)[:5]]) for t in valid_topics]
    })

    # 클러스터별 요약
    cluster_summary = topic_clusters.groupby('cluster').agg({
        'topic_num': lambda x: list(x),
        'cnt': 'sum'
    }).reset_index()

    cluster_summary.columns = ['cluster', 'topic_num', 'cnt']

    print(f"\n[클러스터 요약]")
    print(cluster_summary.to_string(index=False))

    return topic_clusters, cluster_summary