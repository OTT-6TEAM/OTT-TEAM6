import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import time
import re
from tqdm.asyncio import tqdm
import random
from datetime import datetime

# ============================================================
# 설정
# ============================================================
# 동시 요청 수 (너무 높으면 차단될 수 있음)
MAX_CONCURRENT_REQUESTS = 15  # 10-20이 적정

# Rate Limiting (초당 요청 수)
MAX_REQUESTS_PER_SECOND = 5  # 3-5가 안전

# 타임아웃
TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)

# 재시도
MAX_RETRIES = 3


# ============================================================
# Rate Limiter
# ============================================================
class RateLimiter:
    """초당 요청 수를 제한하는 Rate Limiter"""
    def __init__(self, rate):
        self.rate = rate
        self.tokens = rate
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            
            # 토큰 보충
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.updated_at = now

            # 토큰이 부족하면 대기
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 1

            self.tokens -= 1


rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)


# ============================================================
# 비동기 스크래핑 함수
# ============================================================
async def get_imdb_data_async(session, imdb_id, semaphore, max_retries=MAX_RETRIES):
    """
    비동기로 IMDB 데이터를 가져옵니다.
    
    Args:
        session: aiohttp ClientSession
        imdb_id: IMDB ID
        semaphore: 동시 요청 수 제한용 세마포어
        max_retries: 최대 재시도 횟수
    """
    url = f"https://www.imdb.com/title/{imdb_id}/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    async with semaphore:  # 동시 요청 수 제한
        for attempt in range(max_retries):
            try:
                # Rate limiting
                await rate_limiter.acquire()
                
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    html = await response.text()
                    
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # IMDB Rating 추출 (여러 가능한 클래스 시도)
                    imdb_rating = None
                    
                    # 방법 1: 특정 클래스
                    rating_span = soup.find('span', class_='sc-4dc495c1-1')
                    if rating_span:
                        imdb_rating = rating_span.text.strip()
                    
                    # 방법 2: 다른 가능한 클래스
                    if not imdb_rating:
                        rating_span = soup.find('span', {'data-testid': 'rating-value'})
                        if rating_span:
                            imdb_rating = rating_span.text.strip()
                    
                    # 방법 3: 정규표현식으로 찾기
                    if not imdb_rating:
                        rating_pattern = re.compile(r'(\d+\.\d+)/10')
                        match = rating_pattern.search(html)
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
                    
            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return {
                        'imdb_id': imdb_id,
                        'imdb_rating': None,
                        'metascore': None,
                        'status': f'error: {str(e)}',
                        'url': url
                    }
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 2)
                    continue
                else:
                    return {
                        'imdb_id': imdb_id,
                        'imdb_rating': None,
                        'metascore': None,
                        'status': 'error: timeout',
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
        
        # 모든 재시도 실패
        return {
            'imdb_id': imdb_id,
            'imdb_rating': None,
            'metascore': None,
            'status': 'error: max retries exceeded',
            'url': url
        }


# ============================================================
# 배치 처리 함수
# ============================================================
async def process_batch(session, batch_data, semaphore, pbar):
    """
    배치 단위로 데이터를 처리합니다.
    
    Args:
        session: aiohttp ClientSession
        batch_data: 처리할 데이터 리스트
        semaphore: 세마포어
        pbar: 진행 표시줄
    """
    tasks = []
    for row_data in batch_data:
        imdb_id = row_data['imdb_id']
        task = get_imdb_data_async(session, imdb_id, semaphore)
        tasks.append(task)
    
    # 모든 태스크를 동시에 실행
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 처리
    processed_results = []
    for row_data, scraping_result in zip(batch_data, results):
        if isinstance(scraping_result, dict):
            # 원본 데이터와 병합
            result = {
                'imdb_id': row_data['imdb_id'],
                'series_name': row_data.get('name', ''),
                'original_vote_count': row_data.get('vote_count', ''),
                'imdb_rating': scraping_result['imdb_rating'],
                'metascore': scraping_result['metascore'],
                'url': scraping_result['url'],
                'status': scraping_result['status']
            }
            processed_results.append(result)
            pbar.update(1)
        else:
            # 예외 발생
            result = {
                'imdb_id': row_data['imdb_id'],
                'series_name': row_data.get('name', ''),
                'original_vote_count': row_data.get('vote_count', ''),
                'imdb_rating': None,
                'metascore': None,
                'url': f"https://www.imdb.com/title/{row_data['imdb_id']}/",
                'status': f'exception: {str(scraping_result)}'
            }
            processed_results.append(result)
            pbar.update(1)
    
    return processed_results


# ============================================================
# 메인 비동기 함수
# ============================================================
async def main_async(df_filtered):
    """
    메인 비동기 스크래핑 함수
    
    Args:
        df_filtered: 필터링된 DataFrame
    """
    print()
    print("🚀 비동기 스크래핑 시작")
    print(f"⚙️  동시 요청 수: {MAX_CONCURRENT_REQUESTS}")
    print(f"⚙️  초당 요청 수: {MAX_REQUESTS_PER_SECOND}")
    print("-" * 60)
    
    # 세마포어 생성 (동시 요청 수 제한)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # DataFrame을 딕셔너리 리스트로 변환
    data_list = df_filtered.to_dict('records')
    
    # 배치 크기 (중간 저장 단위)
    batch_size = 100
    
    # 결과 저장
    all_results = []
    
    # Connection pooling을 위한 커넥터 설정
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS * 2,  # 커넥션 풀 크기
        limit_per_host=MAX_CONCURRENT_REQUESTS,
        force_close=False,  # 커넥션 재사용
        enable_cleanup_closed=True
    )
    
    # aiohttp ClientSession 생성
    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        # 진행 표시줄
        with tqdm(total=len(data_list), desc="진행 상황", unit="개") as pbar:
            # 배치 단위로 처리
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                # 배치 처리
                batch_results = await process_batch(session, batch, semaphore, pbar)
                all_results.extend(batch_results)
                
                # 중간 저장
                if len(all_results) % batch_size == 0:
                    temp_df = pd.DataFrame(all_results)
                    temp_df.to_csv('imdb_scraping_temp.csv', index=False, encoding='utf-8-sig')
    
    return all_results


# ============================================================
# 메인 함수
# ============================================================
def main():
    print("=" * 60)
    print("IMDB Rating & Metascore 비동기 스크래핑")
    print("⚡ 동기 방식보다 5-10배 빠릅니다!")
    print("=" * 60)
    print()
    
    # 시작 시간
    start_time = datetime.now()
    
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
    
    # 예상 소요 시간 계산 (비동기 방식)
    # 동기: 1.5초 * N개
    # 비동기: (N / 동시요청수) / 초당요청수
    estimated_time_sync = len(df_filtered) * 1.5 / 60
    estimated_time_async = (len(df_filtered) / MAX_CONCURRENT_REQUESTS) / MAX_REQUESTS_PER_SECOND / 60
    
    print(f"예상 소요 시간:")
    print(f"  - 동기 방식: 약 {estimated_time_sync:.1f}분")
    print(f"  - 비동기 방식: 약 {estimated_time_async:.1f}분 ⚡")
    print(f"  - 속도 향상: 약 {estimated_time_sync/estimated_time_async:.1f}배 빠름!")
    print()
    
    response = input("계속 진행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("작업이 취소되었습니다.")
        return None
    
    # 비동기 실행
    try:
        results = asyncio.run(main_async(df_filtered))
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return None
    
    print()
    print("-" * 60)
    
    # 최종 결과를 DataFrame으로 변환
    result_df = pd.DataFrame(results)
    
    # CSV 저장
    output_file = 'imdb_ratings_metascores_async.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 종료 시간
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print()
    print("=" * 60)
    print("스크래핑 완료!")
    print("=" * 60)
    print()
    print(f"✓ 총 {len(result_df)}개의 시리즈 처리")
    print(f"✓ 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"✓ 평균 속도: {len(result_df)/elapsed:.1f}개/초")
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
        for status in error_df['status'].value_counts().head(5).items():
            print(f"  - {status[0][:50]}: {status[1]}개")
    
    print()
    print("=" * 60)
    print("⚡ 성능 비교")
    print("=" * 60)
    sync_time = len(result_df) * 1.5
    speedup = sync_time / elapsed if elapsed > 0 else 0
    print(f"동기 방식 예상 시간: {sync_time:.1f}초 ({sync_time/60:.1f}분)")
    print(f"비동기 방식 실제 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"속도 향상: {speedup:.1f}배 ⚡⚡⚡")
    
    return result_df


if __name__ == "__main__":
    result_df = main()
    
    if result_df is not None:
        print()
        print("=" * 60)
        print("샘플 결과 (처음 10개)")
        print("=" * 60)
        print(result_df[['imdb_id', 'series_name', 'imdb_rating', 'metascore', 'status']].head(10).to_string())
