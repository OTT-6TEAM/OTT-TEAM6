import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import time
import re
from tqdm.asyncio import tqdm
import argparse
from datetime import datetime


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
            
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.updated_at = now

            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 1

            self.tokens -= 1


# ============================================================
# 비동기 스크래핑 함수
# ============================================================
async def get_imdb_data_async(session, imdb_id, semaphore, rate_limiter, max_retries=3):
    """비동기로 IMDB 데이터를 가져옵니다."""
    url = f"https://www.imdb.com/title/{imdb_id}/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                await rate_limiter.acquire()
                
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    html = await response.text()
                    
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # IMDB Rating 추출
                    imdb_rating = None
                    rating_span = soup.find('span', class_='sc-4dc495c1-1')
                    if rating_span:
                        imdb_rating = rating_span.text.strip()
                    
                    if not imdb_rating:
                        rating_span = soup.find('span', {'data-testid': 'rating-value'})
                        if rating_span:
                            imdb_rating = rating_span.text.strip()
                    
                    if not imdb_rating:
                        rating_pattern = re.compile(r'(\d+\.\d+)/10')
                        match = rating_pattern.search(html)
                        if match:
                            imdb_rating = match.group(1)
                    
                    # Metascore 추출
                    metascore = None
                    metascore_span = soup.find('span', class_='sc-9fe7b0ef-0')
                    if metascore_span:
                        metascore = metascore_span.text.strip()
                    
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
                    await asyncio.sleep((attempt + 1) * 2)
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
        
        return {
            'imdb_id': imdb_id,
            'imdb_rating': None,
            'metascore': None,
            'status': 'error: max retries exceeded',
            'url': url
        }


# ============================================================
# 배치 처리
# ============================================================
async def process_batch(session, batch_data, semaphore, rate_limiter, pbar, max_retries):
    """배치 단위로 데이터를 처리합니다."""
    tasks = []
    for row_data in batch_data:
        imdb_id = row_data['imdb_id']
        task = get_imdb_data_async(session, imdb_id, semaphore, rate_limiter, max_retries)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for row_data, scraping_result in zip(batch_data, results):
        if isinstance(scraping_result, dict):
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
async def main_async(df_filtered, config):
    """메인 비동기 스크래핑 함수"""
    print()
    print("🚀 비동기 스크래핑 시작")
    print(f"⚙️  동시 요청 수: {config['concurrent']}")
    print(f"⚙️  초당 요청 수: {config['rate']}")
    print(f"⚙️  재시도 횟수: {config['retries']}")
    print(f"⚙️  배치 크기: {config['batch_size']}")
    print("-" * 60)
    
    semaphore = asyncio.Semaphore(config['concurrent'])
    rate_limiter = RateLimiter(config['rate'])
    
    data_list = df_filtered.to_dict('records')
    all_results = []
    
    connector = aiohttp.TCPConnector(
        limit=config['concurrent'] * 2,
        limit_per_host=config['concurrent'],
        force_close=False,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=config['timeout'])
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with tqdm(total=len(data_list), desc="진행 상황", unit="개") as pbar:
            for i in range(0, len(data_list), config['batch_size']):
                batch = data_list[i:i + config['batch_size']]
                
                batch_results = await process_batch(
                    session, batch, semaphore, rate_limiter, pbar, config['retries']
                )
                all_results.extend(batch_results)
                
                # 중간 저장
                if len(all_results) % config['batch_size'] == 0:
                    temp_df = pd.DataFrame(all_results)
                    temp_df.to_csv('imdb_scraping_temp.csv', index=False, encoding='utf-8-sig')
    
    return all_results


# ============================================================
# 메인 함수
# ============================================================
def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='IMDB 비동기 스크래핑 (고성능)')
    parser.add_argument('--concurrent', '-c', type=int, default=15,
                       help='동시 요청 수 (기본: 15)')
    parser.add_argument('--rate', '-r', type=int, default=5,
                       help='초당 요청 수 (기본: 5)')
    parser.add_argument('--retries', type=int, default=3,
                       help='재시도 횟수 (기본: 3)')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                       help='배치 크기 (기본: 100)')
    parser.add_argument('--timeout', '-t', type=int, default=30,
                       help='타임아웃 초 (기본: 30)')
    parser.add_argument('--input', '-i', default='tv_series_2005_2015_FULL.csv',
                       help='입력 CSV 파일')
    parser.add_argument('--output', '-o', default='imdb_ratings_metascores_async.csv',
                       help='출력 CSV 파일')
    parser.add_argument('--vote-threshold', '-v', type=int, default=30,
                       help='최소 vote_count (기본: 30)')
    
    # 프리셋 옵션
    parser.add_argument('--preset', choices=['safe', 'balanced', 'fast'],
                       help='프리셋 설정 (safe/balanced/fast)')
    
    args = parser.parse_args()
    
    # 프리셋 적용
    if args.preset == 'safe':
        args.concurrent = 5
        args.rate = 2
    elif args.preset == 'balanced':
        args.concurrent = 15
        args.rate = 5
    elif args.preset == 'fast':
        args.concurrent = 25
        args.rate = 8
    
    # 설정 딕셔너리
    config = {
        'concurrent': args.concurrent,
        'rate': args.rate,
        'retries': args.retries,
        'batch_size': args.batch_size,
        'timeout': args.timeout
    }
    
    print("=" * 60)
    print("IMDB Rating & Metascore 비동기 스크래핑 (고급)")
    print("⚡ 설정 가능한 고성능 버전")
    print("=" * 60)
    print()
    
    # 설정 출력
    print("현재 설정:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()
    
    start_time = datetime.now()
    
    # CSV 파일 읽기
    try:
        df_series = pd.read_csv(args.input)
        print(f"✓ CSV 파일 로딩 완료: {args.input}")
        print(f"  - 전체 시리즈: {len(df_series)}개")
    except FileNotFoundError as e:
        print(f"✗ CSV 파일을 찾을 수 없습니다: {e}")
        return None
    
    # 필터링
    df_filtered = df_series[(df_series['vote_count'] >= args.vote_threshold) & 
                           (df_series['imdb_id'].notna())]
    
    print(f"✓ 필터링 완료 (vote_count >= {args.vote_threshold} & imdb_id 존재)")
    print(f"  - 필터링된 시리즈: {len(df_filtered)}개")
    print()
    
    # 예상 시간
    estimated_time = (len(df_filtered) / config['concurrent']) / config['rate'] / 60
    print(f"예상 소요 시간: 약 {estimated_time:.1f}분")
    print()
    
    # 비동기 실행
    try:
        results = asyncio.run(main_async(df_filtered, config))
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return None
    
    print()
    print("-" * 60)
    
    # 결과 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    
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
    print(f"✓ 결과 파일: {args.output}")
    print()
    
    # 통계
    success_count = len(result_df[result_df['status'] == 'success'])
    rating_count = result_df['imdb_rating'].notna().sum()
    metascore_count = result_df['metascore'].notna().sum()
    
    print("=" * 60)
    print("통계")
    print("=" * 60)
    print(f"성공률: {success_count/len(result_df)*100:.1f}% ({success_count}/{len(result_df)})")
    print(f"IMDB Rating: {rating_count}개 ({rating_count/len(result_df)*100:.1f}%)")
    print(f"Metascore: {metascore_count}개 ({metascore_count/len(result_df)*100:.1f}%)")
    
    # 성능 비교
    sync_time = len(result_df) * 1.5
    speedup = sync_time / elapsed if elapsed > 0 else 0
    print()
    print("=" * 60)
    print("⚡ 성능 비교")
    print("=" * 60)
    print(f"동기 방식 예상: {sync_time/60:.1f}분")
    print(f"비동기 실제: {elapsed/60:.1f}분")
    print(f"속도 향상: {speedup:.1f}배 ⚡")
    
    return result_df


if __name__ == "__main__":
    result_df = main()
    
    if result_df is not None:
        print()
        print("=" * 60)
        print("샘플 결과 (처음 5개)")
        print("=" * 60)
        print(result_df[['imdb_id', 'series_name', 'imdb_rating', 'metascore', 'status']].head(5).to_string())
