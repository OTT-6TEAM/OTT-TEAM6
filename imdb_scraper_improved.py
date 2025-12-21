import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from tqdm import tqdm
import random

def get_imdb_data(imdb_id, max_retries=3):
    """
    IMDB ID로부터 rating과 metascore를 추출합니다.
    재시도 로직 포함
    """
    url = f"https://www.imdb.com/title/{imdb_id}/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # IMDB Rating 추출 (여러 가능한 클래스 시도)
            imdb_rating = None
            
            # 방법 1: 특정 클래스
            rating_span = soup.find('span', class_='sc-4dc495c1-1')
            if rating_span:
                imdb_rating = rating_span.text.strip()
            
            # 방법 2: 다른 가능한 클래스 (IMDB가 클래스를 변경할 수 있음)
            if not imdb_rating:
                rating_span = soup.find('span', {'data-testid': 'rating-value'})
                if rating_span:
                    imdb_rating = rating_span.text.strip()
            
            # 방법 3: 정규표현식으로 찾기
            if not imdb_rating:
                rating_pattern = re.compile(r'(\d+\.\d+)/10')
                match = rating_pattern.search(response.text)
                if match:
                    imdb_rating = match.group(1)
            
            # Metascore 추출
            metascore = None
            
            # 방법 1: 특정 클래스
            metascore_span = soup.find('span', class_='sc-9fe7b0ef-0')
            if metascore_span:
                metascore = metascore_span.text.strip()
            
            # 방법 2: metacritic 키워드 검색
            if not metascore:
                metascore_span = soup.find('span', class_=re.compile('metacritic-score'))
                if metascore_span:
                    metascore = metascore_span.text.strip()
            
            return {
                'imdb_id': imdb_id,
                'imdb_rating': imdb_rating,
                'metascore': metascore,
                'status': 'success',
                'url': url
            }
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 2, 4, 6초
                time.sleep(wait_time)
                continue
            else:
                return {
                    'imdb_id': imdb_id,
                    'imdb_rating': None,
                    'metascore': None,
                    'status': f'error: {str(e)}',
                    'url': url
                }
        except Exception as e:
            return {
                'imdb_id': imdb_id,
                'imdb_rating': None,
                'metascore': None,
                'status': f'parsing error: {str(e)}',
                'url': url
            }

def main():
    print("=" * 60)
    print("IMDB Rating & Metascore 스크래핑 시작")
    print("=" * 60)
    print()
    
    # CSV 파일 읽기
    try:
        df_series = pd.read_csv("tv_series_2005_2015_FULL.csv")
        df_seasons = pd.read_csv("tv_seasons_2005_2015_FULL.csv")
        print(f"✓ CSV 파일 로딩 완료")
        print(f"  - 전체 시리즈: {len(df_series)}개")
    except FileNotFoundError as e:
        print(f"✗ CSV 파일을 찾을 수 없습니다: {e}")
        return None
    
    # 조건에 맞는 데이터 필터링
    df_filtered = df_series[(df_series['vote_count'] >= 30) & (df_series['imdb_id'].notna())]
    
    print(f"✓ 필터링 완료 (vote_count >= 30 & imdb_id 존재)")
    print(f"  - 필터링된 시리즈: {len(df_filtered)}개")
    print()
    
    # 예상 소요 시간 계산 (1.5초 딜레이 기준)
    estimated_time = len(df_filtered) * 1.5 / 60  # 분 단위
    print(f"예상 소요 시간: 약 {estimated_time:.1f}분")
    print()
    
    response = input("계속 진행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("작업이 취소되었습니다.")
        return None
    
    print()
    print("스크래핑 시작...")
    print("-" * 60)
    
    # 결과 저장할 리스트
    results = []
    
    # 각 IMDB ID에 대해 스크래핑
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="진행 상황"):
        imdb_id = row['imdb_id']
        
        # 데이터 가져오기
        data = get_imdb_data(imdb_id)
        
        # 원본 데이터와 병합
        result = {
            'imdb_id': imdb_id,
            'series_name': row.get('name', ''),
            'original_vote_count': row.get('vote_count', ''),
            'imdb_rating': data['imdb_rating'],
            'metascore': data['metascore'],
            'url': data['url'],
            'status': data['status']
        }
        
        results.append(result)
        
        # 서버 부담을 줄이기 위한 딜레이 (1-2초 랜덤)
        time.sleep(random.uniform(1.0, 2.0))
        
        # 중간 저장 (50개마다)
        if len(results) % 50 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv('imdb_scraping_temp.csv', index=False, encoding='utf-8-sig')
    
    print()
    print("-" * 60)
    
    # 최종 결과를 DataFrame으로 변환
    result_df = pd.DataFrame(results)
    
    # CSV 저장
    output_file = 'imdb_ratings_metascores.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print()
    print("=" * 60)
    print("스크래핑 완료!")
    print("=" * 60)
    print()
    print(f"✓ 총 {len(result_df)}개의 시리즈 처리")
    print(f"✓ 결과 파일: {output_file}")
    print()
    
    # 상세 통계
    success_count = len(result_df[result_df['status'] == 'success'])
    rating_count = result_df['imdb_rating'].notna().sum()
    metascore_count = result_df['metascore'].notna().sum()
    
    print("=" * 60)
    print("통계")
    print("=" * 60)
    print(f"성공적으로 처리됨:    {success_count:4d}개 ({success_count/len(result_df)*100:.1f}%)")
    print(f"IMDB Rating 있음:     {rating_count:4d}개 ({rating_count/len(result_df)*100:.1f}%)")
    print(f"Metascore 있음:       {metascore_count:4d}개 ({metascore_count/len(result_df)*100:.1f}%)")
    print(f"둘 다 있음:           {result_df[(result_df['imdb_rating'].notna()) & (result_df['metascore'].notna())].shape[0]:4d}개")
    print()
    
    # 에러 분석
    error_df = result_df[result_df['status'] != 'success']
    if len(error_df) > 0:
        print(f"⚠ 에러 발생:          {len(error_df):4d}개")
        print("\n에러 유형:")
        for status in error_df['status'].value_counts().head(3).items():
            print(f"  - {status[0][:50]}: {status[1]}개")
    
    return result_df

if __name__ == "__main__":
    result_df = main()
    
    if result_df is not None:
        print()
        print("=" * 60)
        print("샘플 결과 (처음 10개)")
        print("=" * 60)
        print(result_df[['series_name', 'imdb_rating', 'metascore', 'status']].head(10).to_string())
