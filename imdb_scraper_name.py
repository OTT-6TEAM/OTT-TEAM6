import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from tqdm import tqdm

def get_imdb_data(imdb_id):
    """
    IMDB ID로부터 rating과 metascore를 추출합니다.
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
        imdb_rating = None
        rating_span = soup.find('span', class_='sc-4dc495c1-1')
        if rating_span:
            imdb_rating = rating_span.text.strip()
        
        # Metascore 추출
        metascore = None
        metascore_span = soup.find('span', class_='sc-9fe7b0ef-0')
        if metascore_span:
            metascore = metascore_span.text.strip()
        
        return {
            'imdb_id': imdb_id,
            'imdb_rating': imdb_rating,
            'metascore': metascore,
            'status': 'success'
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {imdb_id}: {str(e)}")
        return {
            'imdb_id': imdb_id,
            'imdb_rating': None,
            'metascore': None,
            'status': f'error: {str(e)}'
        }
    except Exception as e:
        print(f"Parsing error for {imdb_id}: {str(e)}")
        return {
            'imdb_id': imdb_id,
            'imdb_rating': None,
            'metascore': None,
            'status': f'parsing error: {str(e)}'
        }

def main():
    print("데이터 로딩 중...")
    
    # CSV 파일 읽기
    df_series = pd.read_csv("tv_series_2005_2015_FULL.csv")
    df_seasons = pd.read_csv("tv_seasons_2005_2015_FULL.csv")
    
    # 조건에 맞는 데이터 필터링
    df_filtered = df_series[(df_series['vote_count'] >= 30) & (df_series['imdb_id'].notna())]
    
    print(f"필터링된 시리즈 개수: {len(df_filtered)}")
    print(f"스크래핑 시작...\n")
    
    # 결과 저장할 리스트
    results = []
    
    # 각 IMDB ID에 대해 스크래핑
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="스크래핑 진행"):
        imdb_id = row['imdb_id']
        
        # 데이터 가져오기
        data = get_imdb_data(imdb_id)
        
        # 원본 데이터와 병합
        result = {
            'imdb_id': imdb_id,
            'series_name': row.get('name', ''),
            'imdb_rating': data['imdb_rating'],
            'metascore': data['metascore'],
            'status': data['status']
        }
        
        results.append(result)
        
        # 서버 부담을 줄이기 위한 딜레이 (1-2초)
        time.sleep(1.5)
        
        # 중간 저장 (100개마다)
        if len(results) % 100 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv('imdb_scraping_temp.csv', index=False, encoding='utf-8-sig')
            print(f"\n중간 저장 완료: {len(results)}개")
    
    # 최종 결과를 DataFrame으로 변환
    result_df = pd.DataFrame(results)
    
    # CSV 저장
    output_file = 'imdb_ratings_metascores.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n스크래핑 완료!")
    print(f"총 {len(result_df)}개의 시리즈 처리")
    print(f"결과 파일: {output_file}")
    
    # 통계 출력
    print("\n=== 통계 ===")
    print(f"성공: {len(result_df[result_df['status'] == 'success'])}개")
    print(f"IMDB Rating 있음: {result_df['imdb_rating'].notna().sum()}개")
    print(f"Metascore 있음: {result_df['metascore'].notna().sum()}개")
    
    return result_df

if __name__ == "__main__":
    result_df = main()
    print("\n처음 10개 결과:")
    print(result_df.head(10))
