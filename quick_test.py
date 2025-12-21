"""
빠른 테스트 스크립트
===================
전체 스크래핑 전에 몇 개의 샘플로 테스트해보세요.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def test_imdb_scraping(num_samples=5):
    """
    CSV 파일에서 몇 개의 샘플만 테스트합니다.
    """
    print("=" * 60)
    print("IMDB 스크래핑 테스트")
    print("=" * 60)
    print()
    
    try:
        # CSV 파일 읽기
        df_series = pd.read_csv("tv_series_2005_2015_FULL.csv")
        
        # 필터링
        df_filtered = df_series[(df_series['vote_count'] >= 30) & (df_series['imdb_id'].notna())]
        
        # 샘플 선택
        df_sample = df_filtered.head(num_samples)
        
        print(f"✓ 총 {len(df_filtered)}개 중 {num_samples}개 샘플 테스트")
        print()
        
        results = []
        
        for idx, row in df_sample.iterrows():
            imdb_id = row['imdb_id']
            series_name = row.get('name', 'Unknown')
            url = f"https://www.imdb.com/title/{imdb_id}/"
            
            print(f"테스트 중: {series_name} ({imdb_id})")
            print(f"  URL: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Rating 추출
                rating_span = soup.find('span', class_='sc-4dc495c1-1')
                rating = rating_span.text if rating_span else None
                
                # Metascore 추출
                metascore_span = soup.find('span', class_='sc-9fe7b0ef-0')
                metascore = metascore_span.text if metascore_span else None
                
                print(f"  ✓ IMDB Rating: {rating}")
                print(f"  ✓ Metascore: {metascore}")
                print()
                
                results.append({
                    'series_name': series_name,
                    'imdb_id': imdb_id,
                    'rating': rating,
                    'metascore': metascore,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"  ✗ 에러: {str(e)[:100]}")
                print()
                
                results.append({
                    'series_name': series_name,
                    'imdb_id': imdb_id,
                    'rating': None,
                    'metascore': None,
                    'status': f'error: {str(e)[:50]}'
                })
            
            time.sleep(2)  # 지연
        
        # 결과 요약
        print("=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)
        
        result_df = pd.DataFrame(results)
        success = len(result_df[result_df['status'] == 'success'])
        has_rating = result_df['rating'].notna().sum()
        has_metascore = result_df['metascore'].notna().sum()
        
        print(f"성공: {success}/{num_samples}")
        print(f"Rating 획득: {has_rating}/{num_samples}")
        print(f"Metascore 획득: {has_metascore}/{num_samples}")
        print()
        
        if success == num_samples:
            print("✓ 테스트 성공! 전체 스크래핑을 진행할 수 있습니다.")
        elif success > 0:
            print("⚠ 일부만 성공했습니다. 네트워크 상태를 확인하세요.")
        else:
            print("✗ 테스트 실패. 네트워크 또는 HTML 구조를 확인하세요.")
        
        print()
        print("상세 결과:")
        print(result_df.to_string())
        
    except FileNotFoundError:
        print("✗ CSV 파일을 찾을 수 없습니다.")
        print("  tv_series_2005_2015_FULL.csv 파일이 같은 폴더에 있는지 확인하세요.")
    except Exception as e:
        print(f"✗ 예상치 못한 에러: {e}")

if __name__ == "__main__":
    print("\n이 스크립트는 전체 스크래핑 전에 몇 개만 테스트합니다.\n")
    
    # 테스트할 샘플 수 입력
    try:
        num = input("테스트할 샘플 수 (기본값 5): ").strip()
        num = int(num) if num else 5
    except:
        num = 5
    
    print()
    test_imdb_scraping(num)
    
    print()
    print("-" * 60)
    print("테스트가 성공적이면 다음 명령으로 전체 스크래핑을 실행하세요:")
    print("python imdb_scraper_improved.py")
    print("-" * 60)
