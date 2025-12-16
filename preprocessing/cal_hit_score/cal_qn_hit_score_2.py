import numpy as np

def scaler(data, column, min_val, max_val):
    data[f"{column}_scaled"] = (
            (np.log(data[column]) - min_val) / (max_val - min_val)
    ).clip(0, 1)
    return data

def num_votes_scaler(data, min_val = 1, max_val = 17):
    return scaler(data, 'num_votes', min_val, max_val)

def rating_scaler(data, min_val = 1, max_val = 10):
    column = 'rating'
    data[f"{column}_scaled"] = (
            (data[column] - min_val) / (max_val - min_val)
    ).clip(0, 1)
    return data

def cal_qn_hit_score(data):
    W_VOTES = 0.72
    W_RATING = 0.28

    # 가중 평균으로 두 데이터를 통합
    data = data.assign(
        rating = ((data['tmdb_rating'] * data['tmdb_num_votes']) +
                  (data['imdb_rating'] * data['imdb_num_votes']))/
                 (data['imdb_num_votes'] + data['tmdb_num_votes'])
    )
    data = data.assign(num_votes = (data['imdb_num_votes'] + data['tmdb_num_votes']))

    data = num_votes_scaler(data)
    data = rating_scaler(data)

    qn_hit_score = (
        (data["num_votes_scaled"] * W_VOTES) +
        (data["rating_scaled"] * W_RATING)
    )

    return qn_hit_score