import numpy as np

def scaler(data, column, min_val, max_val):
    data[f"{column}_scaled"] = (
            (np.log(data[column]) - min_val) / (max_val - min_val)
    ).clip(0, 1)
    return data

def num_votes_scaler(data, min_val = 1, max_val = 17):
    return scaler(data, 'num_votes', min_val, max_val)

def adjust_rating_scaler(data, min_val = 1, max_val = 10):
    column = 'adjust_rating'
    data[f"{column}_scaled"] = (
            (data[column] - min_val) / (max_val - min_val)
    ).clip(0, 1)
    return data

def cal_qn_hit_score(data, min_ratio = 0.2):
    W_VOTES = 0.74
    W_RATING = 0.26

    # 가중 평균으로 두 데이터를 통합
    data = data.assign(
        rating = ((data['tmdb_rating'] * data['tmdb_num_votes']) +
                  (data['imdb_rating'] * data['imdb_num_votes']))/
                 (data['imdb_num_votes'] + data['tmdb_num_votes'])
    )

    # 보조 지표들 산출
    data = data.assign(num_votes = (data['imdb_num_votes'] + data['tmdb_num_votes']))
    rating_avg = (data['num_votes'] * data['imdb_rating']).sum() / data['num_votes'].sum()
    min_vote = np.quantile(data['num_votes'], q=min_ratio)

    # 최종 weighted_ratin 계산
    data = data.assign(
        weighted_rating = (data['num_votes'] / (data['num_votes'] + min_vote)) *data['rating'] +
                        (min_vote / (data['num_votes'] + min_vote)) * rating_avg
    )

    alpha = 0.2 + 0.6 * ((data['rating'] - 1) / (10 - 1))
    data = data.assign(adjust_rating=data['weighted_rating'] * alpha + data['rating'] * (1 - alpha))

    data = num_votes_scaler(data)
    data = adjust_rating_scaler(data)

    qn_hit_score = (
        (data["num_votes_scaled"] * W_VOTES) +
        (data["adjust_rating_scaled"] * W_RATING)
    )
    return qn_hit_score