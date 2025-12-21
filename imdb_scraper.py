# ==========================================================
# IMDB 데이터 크롤러 (리뷰, 평점, 메타스코어)
# TV 시리즈 전용 - 비동기 + Rate Limiting
# ==========================================================

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import json
import re
import time
from pathlib import Path
import random

# ==========================================================
# 설정
# ==========================================================

# Rate Limiting (IMDB는 엄격하므로 보수적으로 설정)
MAX_CALLS_PER_SECOND = 2  # 초당 2회 (안전한 속도)
TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)
MAX_RETRIES = 3
RETRY_DELAY = [2, 5, 10]  # 재시도 간격 (초)

# User-Agent (실제 브라우저처럼 보이도록)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
]

# 출력 파일
OUTPUT_CSV = "imdb_data_collected.csv"
OUTPUT_PARQUET = "imdb_data_collected.parquet"
CHECKPOINT_FILE = "imdb_checkpoint.json"

# 통계
stats = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "requests": 0,
    "start_time": None
}

# ==========================================================
# Rate Limiter
# ==========================================================
class RateLimiter:
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
            
            # 토큰 부족시 대기
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 1
            
            self.tokens -= 1

rate_limiter = RateLimiter(MAX_CALLS_PER_SECOND)

# ==========================================================
# HTML 가져오기
# ==========================================================
async def fetch_html(session, url, retry=0):
    """HTML 페이지 가져오기 (재시도 로직 포함)"""
    if retry >= MAX_RETRIES:
        stats["failed"] += 1
        return None
    
    await rate_limiter.acquire()
    stats["requests"] += 1
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        async with session.get(url, headers=headers, timeout=TIMEOUT) as resp:
            # Rate limit 감지
            if resp.status == 429 or resp.status == 503:
                wait_time = RETRY_DELAY[min(retry, len(RETRY_DELAY)-1)]
                print(f"⚠️  Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await fetch_html(session, url, retry + 1)
            
            # 404는 정상 케이스 (데이터 없음)
            if resp.status == 404:
                return None
            
            # 기타 에러
            if resp.status != 200:
                if retry < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY[retry])
                    return await fetch_html(session, url, retry + 1)
                return None
            
            html = await resp.text()
            return html
    
    except asyncio.TimeoutError:
        print(f"⚠️  Timeout: {url}")
        if retry < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY[retry])
            return await fetch_html(session, url, retry + 1)
        return None
    
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        if retry < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY[retry])
            return await fetch_html(session, url, retry + 1)
        return None

# ==========================================================
# IMDB 평점 & 메타스코어 추출
# ==========================================================
def extract_rating_and_metascore(soup, imdb_id):
    """메인 페이지에서 평점과 메타스코어 추출"""
    result = {
        'imdb_id': imdb_id,
        'imdb_rating': None,
        'imdb_rating_count': None,
        'meta_score': None
    }
    
    try:
        # IMDB Rating - JSON-LD에서 추출 (가장 정확)
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and 'aggregateRating' in data:
                    rating_data = data['aggregateRating']
                    result['imdb_rating'] = float(rating_data.get('ratingValue', 0))
                    result['imdb_rating_count'] = int(rating_data.get('ratingCount', 0))
                    break
            except:
                continue
        
        # 대체 방법: div[data-testid="hero-rating-bar__aggregate-rating__score"]
        if result['imdb_rating'] is None:
            rating_elem = soup.find('div', {'data-testid': 'hero-rating-bar__aggregate-rating__score'})
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                match = re.search(r'([\d.]+)', rating_text)
                if match:
                    result['imdb_rating'] = float(match.group(1))
        
        # Meta Score - 여러 선택자 시도
        metascore_selectors = [
            {'class': 'metacritic-score-box'},
            {'data-testid': 'metacritic-score'},
            {'class': 'score-meta'}
        ]
        
        for selector in metascore_selectors:
            meta_elem = soup.find('span', selector) or soup.find('div', selector)
            if meta_elem:
                meta_text = meta_elem.get_text(strip=True)
                match = re.search(r'(\d+)', meta_text)
                if match:
                    result['meta_score'] = int(match.group(1))
                    break
    
    except Exception as e:
        print(f"⚠️  Error parsing rating/metascore for {imdb_id}: {e}")
    
    return result

# ==========================================================
# IMDB 리뷰 추출
# ==========================================================
def extract_reviews(soup, imdb_id, max_reviews=10):
    """리뷰 페이지에서 리뷰 추출"""
    reviews = []
    
    try:
        # 리뷰 컨테이너 찾기
        review_containers = soup.find_all('div', {'class': 'review-container'})
        
        for container in review_containers[:max_reviews]:
            review = {}
            
            # 리뷰 제목
            title_elem = container.find('a', {'class': 'title'})
            if title_elem:
                review['title'] = title_elem.get_text(strip=True)
            
            # 평점
            rating_elem = container.find('span', {'class': 'rating-other-user-rating'})
            if rating_elem:
                rating_span = rating_elem.find('span')
                if rating_span:
                    try:
                        review['user_rating'] = int(rating_span.get_text(strip=True))
                    except:
                        pass
            
            # 리뷰 내용
            content_elem = container.find('div', {'class': 'text'})
            if content_elem:
                review['content'] = content_elem.get_text(strip=True)
            
            # 작성자
            author_elem = container.find('span', {'class': 'display-name-link'})
            if author_elem:
                review['author'] = author_elem.get_text(strip=True)
            
            # 날짜
            date_elem = container.find('span', {'class': 'review-date'})
            if date_elem:
                review['date'] = date_elem.get_text(strip=True)
            
            # Helpful 투표
            helpful_elem = container.find('div', {'class': 'actions'})
            if helpful_elem:
                helpful_text = helpful_elem.get_text()
                match = re.search(r'(\d+)\s+out of\s+(\d+)', helpful_text)
                if match:
                    review['helpful'] = f"{match.group(1)}/{match.group(2)}"
            
            if review:  # 최소한 하나의 필드라도 있으면 추가
                reviews.append(review)
        
    except Exception as e:
        print(f"⚠️  Error parsing reviews for {imdb_id}: {e}")
    
    return reviews

# ==========================================================
# 단일 IMDB ID 처리
# ==========================================================
async def scrape_imdb_data(session, imdb_id, series_title=""):
    """하나의 IMDB ID에 대한 모든 데이터 수집"""
    stats["total"] += 1
    
    result = {
        'imdb_id': imdb_id,
        'series_title': series_title,
        'imdb_rating': None,
        'imdb_rating_count': None,
        'meta_score': None,
        'reviews_json': None,
        'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # 1. 메인 페이지에서 평점 & 메타스코어
        main_url = f"https://www.imdb.com/title/{imdb_id}/"
        main_html = await fetch_html(session, main_url)
        
        if main_html:
            soup = BeautifulSoup(main_html, 'html.parser')
            rating_data = extract_rating_and_metascore(soup, imdb_id)
            result.update(rating_data)
        
        # 2. 리뷰 페이지에서 리뷰 수집
        reviews_url = f"https://www.imdb.com/title/{imdb_id}/reviews/"
        reviews_html = await fetch_html(session, reviews_url)
        
        if reviews_html:
            soup_reviews = BeautifulSoup(reviews_html, 'html.parser')
            reviews = extract_reviews(soup_reviews, imdb_id)
            if reviews:
                result['reviews_json'] = json.dumps(reviews, ensure_ascii=False)
        
        stats["success"] += 1
        return result
    
    except Exception as e:
        print(f"❌ Error processing {imdb_id}: {e}")
        stats["failed"] += 1
        return result

# ==========================================================
# 체크포인트 관리
# ==========================================================
def save_checkpoint(processed_ids):
    """진행 상황 저장"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'processed_ids': list(processed_ids)}, f)

def load_checkpoint():
    """이전 진행 상황 로드"""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return set(json.load(f)['processed_ids'])
    return set()

# ==========================================================
# 메인 실행 함수
# ==========================================================
async def main(input_csv_path):
    """
    TMDB 데이터에서 조건에 맞는 시리즈의 IMDB 데이터 수집
    
    Args:
        input_csv_path: TMDB TV 시리즈 CSV 파일 경로
    """
    print("=" * 90)
    print("🎬 IMDB 데이터 크롤러 시작")
    print("=" * 90)
    
    stats["start_time"] = datetime.now()
    t0 = datetime.now()
    
    # 1. 데이터 로드 및 필터링
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv(input_csv_path)
    print(f"✅ 전체 시리즈: {len(df):,}개")
    
    # 조건 필터링: vote_count >= 30 AND imdb_id가 존재
    df_filtered = df[(df['vote_count'] >= 30) & (df['imdb_id'].notna())]
    print(f"✅ 필터링된 시리즈 (vote_count>=30 & imdb_id 존재): {len(df_filtered):,}개")
    
    if len(df_filtered) == 0:
        print("⚠️  조건을 만족하는 데이터가 없습니다.")
        return
    
    # IMDB ID 리스트 추출
    imdb_data = df_filtered[['id', 'title', 'imdb_id']].to_dict('records')
    
    # 체크포인트 로드
    processed_ids = load_checkpoint()
    if processed_ids:
        print(f"📌 이전 진행 상황 로드: {len(processed_ids):,}개 처리 완료")
        imdb_data = [x for x in imdb_data if x['imdb_id'] not in processed_ids]
        print(f"📌 남은 작업: {len(imdb_data):,}개")
    
    if len(imdb_data) == 0:
        print("✅ 모든 데이터가 이미 처리되었습니다.")
        return
    
    # 2. 크롤링 시작
    print(f"\n🚀 크롤링 시작: {len(imdb_data):,}개 시리즈")
    print(f"⚙️  속도 제한: {MAX_CALLS_PER_SECOND}회/초")
    print(f"⏱️  예상 소요 시간: {len(imdb_data)*2/MAX_CALLS_PER_SECOND/60:.1f}분")
    
    # 세션 설정
    connector = aiohttp.TCPConnector(
        limit=10,  # 동시 연결 수 제한
        force_close=True,
        enable_cleanup_closed=True
    )
    
    results = []
    batch_size = 50  # 배치 크기
    
    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        for i in range(0, len(imdb_data), batch_size):
            batch = imdb_data[i:i+batch_size]
            
            # 배치 처리
            batch_results = await asyncio.gather(
                *[scrape_imdb_data(session, item['imdb_id'], item['title']) for item in batch],
                return_exceptions=True
            )
            
            # 결과 수집
            for r in batch_results:
                if isinstance(r, dict):
                    results.append(r)
                    processed_ids.add(r['imdb_id'])
            
            # 주기적으로 저장
            if len(results) % 20 == 0:
                df_results = pd.DataFrame(results)
                df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
                save_checkpoint(processed_ids)
            
            # 진행 상황 출력
            elapsed = (datetime.now() - t0).total_seconds()
            progress = (i + len(batch)) / len(imdb_data) * 100
            rate = stats["total"] / elapsed if elapsed > 0 else 0
            eta = (len(imdb_data) - stats["total"]) / rate / 60 if rate > 0 else 0
            
            print(f"📊 진행: {stats['total']:,}/{len(imdb_data):,} ({progress:.1f}%) | "
                  f"성공: {stats['success']:,} | 실패: {stats['failed']} | "
                  f"속도: {rate:.2f}/s | ETA: {eta:.1f}분")
    
    # 3. 최종 저장
    print("\n💾 최종 저장 중...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    try:
        df_results.to_parquet(OUTPUT_PARQUET, index=False)
    except:
        print("⚠️  Parquet 저장 실패 (CSV만 저장됨)")
    
    # 체크포인트 삭제
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
    
    # 4. 통계 출력
    elapsed = (datetime.now() - t0).total_seconds() / 60
    
    print("\n" + "=" * 90)
    print("🎉 크롤링 완료!")
    print("=" * 90)
    print(f"📌 총 처리: {stats['total']:,}개")
    print(f"📌 성공: {stats['success']:,}개 ({stats['success']/stats['total']*100:.1f}%)")
    print(f"📌 실패: {stats['failed']}개")
    print(f"📌 총 요청: {stats['requests']:,}회")
    print(f"⏱️  총 소요 시간: {elapsed:.1f}분 ({elapsed/60:.2f}시간)")
    print(f"📊 평균 속도: {stats['success']/elapsed:.1f}개/분")
    print("=" * 90)
    
    # 샘플 데이터 표시
    print("\n📊 샘플 데이터:")
    sample = df_results[df_results['imdb_rating'].notna()].head(3)
    for idx, row in sample.iterrows():
        print(f"\n제목: {row['series_title']}")
        print(f"  IMDB ID: {row['imdb_id']}")
        print(f"  IMDB 평점: {row['imdb_rating']}/10 ({row['imdb_rating_count']:,}표)")
        print(f"  Meta Score: {row['meta_score']}")
        if row['reviews_json']:
            reviews = json.loads(row['reviews_json'])
            print(f"  리뷰 수: {len(reviews)}개")
    
    print(f"\n✅ 결과 파일: {OUTPUT_CSV}")

# ==========================================================
# 실행
# ==========================================================
if __name__ == "__main__":
    # 사용 예시
    input_file = "tv_series_2013_0101_0215_FULL.csv"
    
    if not Path(input_file).exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        print("📝 TMDB 수집 스크립트를 먼저 실행하세요.")
    else:
        asyncio.run(main(input_file))
