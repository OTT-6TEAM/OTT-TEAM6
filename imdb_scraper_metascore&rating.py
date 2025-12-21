import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import re

def get_imdb_data(imdb_id):
    """
    IMDB ID로부터 rating과 Metascore를 추출
    """
    url = f"https://www.imdb.com/title/{imdb_id}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # IMDB Rating 추출
        rating = None
        # 새로운 IMDB 디자인의 rating 선택자
        rating_element = soup.find('span', {'class': 'sc-eb51e184-1'})
        if not rating_element:
            # 대체 선택자들
            rating_element = soup.find('span', {'data-testid': 'ratingScore'})
        if rating_element:
            rating_text = rating_element.get_text()
            # "7.5/10" 형태에서 숫자 추출
            match = re.search(r'(\d+\.?\d*)', rating_text)
            if match:
                rating = float(match.group(1))
        
        # Metascore 추출
        metascore = None
        # Metascore 선택자
        metascore_element = soup.find('span', {'class': 'metacritic-score-box'})
        if not metascore_element:
            # 대체 선택자
            metascore_element = soup.find('span', text=re.compile('Metascore'))
            if metascore_element:
                parent = metascore_element.find_parent()
                if parent:
                    score_span = parent.find('span', {'class': 'score'})
                    if score_span:
                        metascore = int(score_span.get_text().strip())
        else:
            metascore = int(metascore_element.get_text().strip())
        
        return {
            'imdb_id': imdb_id,
            'imdb_rating': rating,
            'metascore': metascore,
            'url': url,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Error scraping {imdb_id}: {str(e)}")
        return {
            'imdb_id': imdb_id,
            'imdb_rating': None,
            'metascore': None,
            'url': url,
            'status': f'error: {str(e)}'
        }

def main():
    # CSV 파일 로드
    print("Loading CSV files...")
    df_series = pd.read_csv("tv_series_2005_2015_FULL.csv")
    
    # 조건에 맞는 데이터 필터링
    print("Filtering data...")
    filtered_df = df_series[(df_series['vote_count'] >= 30) & (df_series['imdb_id'].notna())]
    print(f"Found {len(filtered_df)} series matching criteria")
    
    # IMDB ID 리스트
    imdb_ids = filtered_df['imdb_id'].unique().tolist()
    print(f"Total unique IMDB IDs: {len(imdb_ids)}")
    
    # 크롤링 결과 저장
    results = []
    
    print("Starting web scraping...")
    for imdb_id in tqdm(imdb_ids):
        result = get_imdb_data(imdb_id)
        results.append(result)
        
        # IMDB 서버에 부담을 주지 않기 위해 딜레이 추가
        time.sleep(1)  # 1초 대기
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    
    # 원본 데이터와 병합
    merged_df = filtered_df.merge(results_df, on='imdb_id', how='left')
    
    # CSV로 저장
    output_file = 'tv_series_with_ratings.csv'
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    
    # 통계 출력
    print("\n=== Summary Statistics ===")
    print(f"Total series processed: {len(results_df)}")
    print(f"Successfully scraped: {(results_df['status'] == 'success').sum()}")
    print(f"Rating found: {results_df['imdb_rating'].notna().sum()}")
    print(f"Metascore found: {results_df['metascore'].notna().sum()}")
    
    # 결과만 별도로 저장
    results_df.to_csv('imdb_ratings_only.csv', index=False, encoding='utf-8-sig')
    print(f"\nRatings-only file saved to: imdb_ratings_only.csv")

if __name__ == "__main__":
    main()
