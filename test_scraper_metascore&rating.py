import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def get_imdb_data(imdb_id):
    """IMDB ID로부터 rating과 Metascore를 추출"""
    url = f"https://www.imdb.com/title/{imdb_id}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        print(f"Fetching {url}...")
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # IMDB Rating 추출 (여러 방법 시도)
        rating = None
        
        # 방법 1: data-testid 속성
        rating_elem = soup.find('div', {'data-testid': 'hero-rating-bar__aggregate-rating__score'})
        if rating_elem:
            rating_span = rating_elem.find('span', {'class': 'sc-eb51e184-1'})
            if rating_span:
                rating_text = rating_span.get_text()
                rating = float(rating_text.split('/')[0])
        
        # 방법 2: 클래스로 찾기
        if rating is None:
            rating_elem = soup.find('span', {'data-testid': 'rating-button__aggregate-rating__score'})
            if rating_elem:
                rating = float(rating_elem.get_text().split('/')[0])
        
        # Metascore 추출
        metascore = None
        metascore_elem = soup.find('span', {'class': 'score-meta'})
        if metascore_elem:
            metascore = int(metascore_elem.get_text().strip())
        
        print(f"  → Rating: {rating}, Metascore: {metascore}")
        
        return {
            'imdb_id': imdb_id,
            'imdb_rating': rating,
            'metascore': metascore,
            'url': url
        }
        
    except Exception as e:
        print(f"  → Error: {e}")
        return {
            'imdb_id': imdb_id,
            'imdb_rating': None,
            'metascore': None,
            'url': url
        }

# CSV 파일 로드
print("Loading CSV...")
df_series = pd.read_csv("tv_series_2005_2015_FULL.csv")

# 조건 필터링
filtered_df = df_series[(df_series['vote_count'] >= 30) & (df_series['imdb_id'].notna())]
print(f"Total series: {len(filtered_df)}")

# 테스트: 처음 5개만 크롤링
test_ids = filtered_df['imdb_id'].head(5).tolist()
print(f"\nTesting with first 5 IDs: {test_ids}\n")

results = []
for imdb_id in test_ids:
    result = get_imdb_data(imdb_id)
    results.append(result)
    time.sleep(2)  # 2초 대기

# 결과 저장
results_df = pd.DataFrame(results)
results_df.to_csv('test_results.csv', index=False, encoding='utf-8-sig')

print(f"\n✓ Test results saved to: test_results.csv")
print("\nResults:")
print(results_df)
