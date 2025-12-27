from set_up import *

def preprocessing_movie(data):
    """
    영화 데이터 전처리

    Parameters:
    -----------
    data : pd.DataFrame
        원본 영화 데이터

    Returns:
    --------
    tuple : (main_data, roi_data, production_countries, genres, providers_flatrate)
        - main_data: 메인 영화 데이터
        - roi_data: ROI 계산용 데이터 (budget, revenue)
        - production_countries: 제작 국가 정규화 테이블
        - genres: 장르 정규화 테이블
        - providers_flatrate: OTT 플랫폼 정규화 테이블
    """

    # ============================================
    # 1. 기본 필터링
    # ============================================
    data = data[data['vote_count'] >= 30].copy()

    # ============================================
    # 2. 메인 데이터 생성
    # ============================================
    main_data = data[[
        'id', 'imdb_id', 'title', 'original_language',
        'overview', 'release_date', 'runtime', 'genres',
        'keywords', 'poster_path', 'vote_average', 'vote_count',
        'imdb_rating', 'imdb_num_votes'
    ]].rename(
        columns={'vote_average': 'tmdb_rating', 'vote_count': 'tmdb_num_votes'}
    ).dropna().copy()

    # 날짜 형식 변환
    main_data['release_date'] = pd.to_datetime(main_data['release_date'])

    # 런타임 필터링 (45분 ~ 300분)
    main_data = main_data[
        (main_data['runtime'] > 45) & (main_data['runtime'] <= 300)
        ]

    # 언어 처리: 상위 10개 외 언어는 'xx'로 통일
    top_10_languages = main_data['original_language'].value_counts()[:10].index
    main_data.loc[
        ~main_data['original_language'].isin(top_10_languages),
        'original_language'
    ] = 'xx'

    # ============================================
    # 3. ROI 데이터 생성
    # ============================================
    roi_data = data.loc[
        (data['budget'] != 0) & (data['revenue'] != 0),
        ['id', 'imdb_id', 'budget', 'revenue']
    ].dropna().copy()

    # ============================================
    # 4. 정규화 테이블 생성 (파싱 필요한 컬럼들)
    # ============================================

    # 데이터 복사본 생성 (원본 보존)
    data_for_parsing = data.copy()

    # 4-1. production_countries, genre_ids 파싱
    parsing_col = ['production_countries', 'genre_ids']
    parsing_columns(data_for_parsing, parsing_col)

    # 4-2. providers_flatrate 파싱
    data_for_parsing['providers_flatrate'] = optimized_provider_parse(
        data_for_parsing, "providers_flatrate"
    )
    parsing_col.append('providers_flatrate')

    # 4-3. 정규화 테이블 생성
    production_countries, genre_ids, providers_flatrate = table_normalization(
        data_for_parsing, parsing_col
    )

    # ============================================
    # 5. production_countries 전처리
    # ============================================
    production_countries = production_countries.dropna()

    # 상위 10개 국가 외에는 'Other'로 통일
    production_countries_cnt = production_countries['production_countries'].value_counts()
    production_countries.loc[
        production_countries['production_countries'].isin(
            production_countries_cnt[10:].index
        ),
        'production_countries'
    ] = 'Other'

    # 소문자 변환 및 중복 제거
    production_countries['production_countries'] = (
        production_countries['production_countries'].str.lower()
    )
    production_countries = production_countries.drop_duplicates()

    # ============================================
    # 6. genres 전처리 (TMDB API 활용)
    # ============================================
    genre_ids['genre_id'] = genre_ids['genre_id'].astype('int')

    # TMDB API에서 장르 매핑 정보 가져오기
    API_KEY = os.getenv("TMDB_API_KEY")

    if API_KEY:
        genres_df = get_genre_mapping(media_type='movie')

        if not genres_df.empty:
            genres = pd.merge(genre_ids, genres_df, on='genre_id', how='left')
        else:
            print("장르 API 호출 실패. genre_ids만 반환합니다.")
            genres = genre_ids
    else:
        print("TMDB_API_KEY가 설정되지 않았습니다. genre_ids만 반환합니다.")
        genres = genre_ids

    # ============================================
    # 7. providers_flatrate 전처리
    # ============================================
    providers_flatrate = preprocess_providers(providers_flatrate)

    return main_data, roi_data, production_countries, genres, providers_flatrate


def preprocess_providers(providers_flatrate):
    """
    OTT 플랫폼 데이터 전처리

    Parameters:
    -----------
    providers_flatrate : pd.DataFrame
        플랫폼 정규화 테이블

    Returns:
    --------
    pd.DataFrame
        전처리된 플랫폼 테이블
    """

    # 소문자 변환 및 공백 제거
    providers_flatrate['providers_flatrate'] = (
        providers_flatrate['providers_flatrate'].str.lower().str.strip()
    )

    # 불필요한 문자열 일괄 제거
    replacements = {
        ' amazon channel': '',
        ' apple tv channel': '',
        ' amzon channel': '',
        ' on u-next': '',
        ' roku premium channel': ''
    }

    for old, new in replacements.items():
        providers_flatrate['providers_flatrate'] = (
            providers_flatrate['providers_flatrate'].str.replace(old, new, regex=False)
        )

    # '+' → ' plus' 변환
    providers_flatrate['providers_flatrate'] = (
        providers_flatrate['providers_flatrate'].str.replace('+', ' plus', regex=False)
    )

    # 공백 재정리 및 중복 제거
    providers_flatrate['providers_flatrate'] = providers_flatrate['providers_flatrate'].str.strip()
    providers_flatrate = providers_flatrate.drop_duplicates()

    # 플랫폼 이름 통일 (딕셔너리 방식)
    platform_mapping = {
        # Paramount+
        'paramount plus premium': 'paramount plus',
        'paramount plus basic with ads': 'paramount plus',
        'paramount plus mtv': 'paramount plus',
        'paramount plus with showtime': 'paramount plus',
        'paramount plus originals': 'paramount plus',

        # Netflix
        'netflix standard with ads': 'netflix',
        'netflix kids': 'netflix',

        # Movistar
        'movistar plus plus ficción total': 'movistar plus',
        'movistar plus plus': 'movistar plus',
        'movistartv': 'movistar',
        'movistar plus': 'movistar',

        # Amazon Prime
        'amazon prime video with ads': 'amazon prime video',

        # Peacock
        'peacock premium plus': 'peacock plus',

        # YouTube
        'youtube tv': 'youtube',
        'youtube premium': 'youtube',

        # StudioCanal
        'studiocanal presents allstars': 'studiocanal presents',
        'studiocanal presents moviecult': 'studiocanal presents',

        # TV 2
        'tv 2 play': 'tv 2',

        # BBC
        'bbc kids': 'bbc',
        'bbc america': 'bbc',
        'bbc iplayer': 'bbc',
        'bbc player': 'bbc',

        # Discovery
        'discovery  plus': 'discovery plus',

        # AMC
        'amc plus': 'amc',
        'amc channels': 'amc',

        # Lionsgate
        'lionsgate pluss': 'lionsgate',
        'lionsgate play': 'lionsgate',

        # Vix
        'vix gratis': 'vix',
        'vix premium': 'vix',

        # Atres
        'atresplayer': 'atres player',

        # Now TV
        'now tv cinema': 'now tv',

        # Netzkino
        'netzkino select': 'netzkino',

        # Filmtastic
        'filmtastic bei canal plus': 'filmtastic',

        # Starz
        'starzplay': 'starz',

        # Hallmark
        'hallmark tv': 'hallmark',
        'hallmark plus': 'hallmark',

        # Arrow
        'arrow video': 'arrow',

        # Acorn TV
        'acorn tv apple tv': 'acorn tv',
        'acorntv': 'acorn tv',

        # Sky
        'sky go': 'sky',
        'sky x': 'sky',
        'sky store': 'sky',

        # RTL
        'rtl plus max': 'rtl plus',

        # Stingray
        'qello concerts by stingray': 'stingray',
        'stingray all good vibes': 'stingray',
        'stingray classica': 'stingray',
        'stingray karaoke': 'stingray',

        # MGM
        'mgm plus': 'mgm',
    }

    providers_flatrate['providers_flatrate'] = (
        providers_flatrate['providers_flatrate'].replace(platform_mapping)
    )
    providers_flatrate = providers_flatrate.drop_duplicates()

    # 상위 10개 플랫폼 외에는 'other'로 통일
    top_10_providers = providers_flatrate['providers_flatrate'].value_counts()[:10].index
    providers_flatrate.loc[
        ~providers_flatrate['providers_flatrate'].isin(top_10_providers) &
        (providers_flatrate['providers_flatrate'].notnull()),
        'providers_flatrate'
    ] = 'other'

    providers_flatrate = providers_flatrate.drop_duplicates()

    return providers_flatrate