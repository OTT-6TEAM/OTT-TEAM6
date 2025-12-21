# ==========================================================
# IMDB 전체 리뷰 크롤러 - 비동기 최적화 버전
# Pagination Key를 이용한 전체 리뷰 수집 + 비동기 처리
# ==========================================================

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import random
import json

# ==========================================================
# 설정
# ==========================================================

# Rate Limiting (IMDB는 엄격하므로 보수적으로)
MAX_CALLS_PER_SECOND = 2
TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)
MAX_RETRIES = 3

# User-Agent Pool
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
]

# 출력 파일
OUTPUT_CSV = "imdb_reviews_full_async.csv"
OUTPUT_PARQUET = "imdb_reviews_full_async.parquet"
CHECKPOINT_FILE = "imdb_reviews_checkpoint.json"

# 통계
stats = {
    "series_total": 0,
    "series_success": 0,
    "series_failed": 0,
    "reviews_total": 0,
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
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.updated_at = now
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 1
            
            self.tokens -= 1

rate_limiter = RateLimiter(MAX_CALLS_PER_SECOND)

# ==========================================================
# HTML 가져오기
# ==========================================================
async def fetch_html(session, url, retry=0, method='GET', data=None):
    """HTML 페이지 가져오기 (GET/POST 지원)"""
    if retry >= MAX_RETRIES:
        return None
    
    await rate_limiter.acquire()
    stats["requests"] += 1
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
    }
    
    try:
        if method == 'POST':
            async with session.post(url, headers=headers, data=data, timeout=TIMEOUT) as resp:
                return await handle_response(session, url, resp, retry, method, data)
        else:
            async with session.get(url, headers=headers, timeout=TIMEOUT) as resp:
                return await handle_response(session, url, resp, retry, method, data)
    
    except asyncio.TimeoutError:
        if retry < MAX_RETRIES - 1:
            await asyncio.sleep(2 ** retry)
            return await fetch_html(session, url, retry + 1, method, data)
        return None
    
    except Exception as e:
        if retry < MAX_RETRIES - 1:
            await asyncio.sleep(2 ** retry)
            return await fetch_html(session, url, retry + 1, method, data)
        return None

async def handle_response(session, url, resp, retry, method, data):
    """응답 처리"""
    if resp.status == 429 or resp.status == 503:
        wait_time = 5 * (retry + 1)
        print(f"⚠️  Rate limited, waiting {wait_time}s...")
        await asyncio.sleep(wait_time)
        return await fetch_html(session, url, retry + 1, method, data)
    
    if resp.status == 404:
        return None
    
    if resp.status != 200:
        if retry < MAX_RETRIES - 1:
            await asyncio.sleep(2 ** retry)
            return await fetch_html(session, url, retry + 1, method, data)
        return None
    
    return await resp.text()

# ==========================================================
# 리뷰 파싱
# ==========================================================
def parse_review_block(soup, imdb_id):
    """soup에서 리뷰 리스트 파싱"""
    review_blocks = soup.select(".review-container")
    reviews = []
    
    for block in review_blocks:
        # 제목
        title_elem = block.select_one(".title")
        title = title_elem.get_text(strip=True) if title_elem else None
        
        # 내용
        content_elem = block.select_one(".text.show-more__control")
        if not content_elem:
            content_elem = block.select_one(".text")
        content = content_elem.get_text(strip=True) if content_elem else None
        
        # 평점
        rating_elem = block.select_one(".rating-other-user-rating span")
        rating = rating_elem.get_text(strip=True) if rating_elem else None
        
        # 작성자
        author_elem = block.select_one(".display-name-link a")
        if not author_elem:
            author_elem = block.select_one(".display-name-link")
        author = author_elem.get_text(strip=True) if author_elem else None
        
        # 날짜
        date_elem = block.select_one(".review-date")
        date = date_elem.get_text(strip=True) if date_elem else None
        
        # Helpful 투표
        helpful_elem = block.select_one(".actions")
        helpful = None
        if helpful_elem:
            import re
            helpful_text = helpful_elem.get_text()
            match = re.search(r'(\d+)\s+out of\s+(\d+)', helpful_text)
            if match:
                helpful = f"{match.group(1)}/{match.group(2)}"
        
        # Spoiler 여부
        spoiler = "spoiler-warning" in str(block)
        
        reviews.append({
            "imdb_id": imdb_id,
            "review_title": title,
            "review_text": content,
            "review_rating": rating,
            "review_author": author,
            "review_date": date,
            "helpful_votes": helpful,
            "is_spoiler": spoiler
        })
    
    return reviews

# ==========================================================
# 전체 리뷰 수집 (Pagination 지원)
# ==========================================================
async def fetch_all_reviews_for_series(session, imdb_id, series_title="", max_pages=None):
    """
    한 시리즈의 모든 리뷰 수집 (paginationKey 이용)
    
    Args:
        max_pages: 최대 페이지 수 제한 (None이면 전체)
    """
    base_url = f"https://www.imdb.com/title/{imdb_id}/reviews"
    ajax_url = f"https://www.imdb.com/title/{imdb_id}/reviews/_ajax"
    
    all_reviews = []
    page_count = 0
    
    try:
        # 1. 첫 페이지 (GET)
        html = await fetch_html(session, base_url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        all_reviews.extend(parse_review_block(soup, imdb_id))
        page_count += 1
        
        # 2. Pagination Key 찾기
        load_more = soup.select_one("div.load-more-data")
        if not load_more:
            return all_reviews
        
        pagination_key = load_more.get("data-key")
        
        # 3. Ajax 페이지 순회 (POST)
        while pagination_key:
            if max_pages and page_count >= max_pages:
                break
            
            # POST 요청
            payload = {"paginationKey": pagination_key}
            html = await fetch_html(session, ajax_url, method='POST', data=payload)
            
            if not html:
                break
            
            ajax_soup = BeautifulSoup(html, 'html.parser')
            new_reviews = parse_review_block(ajax_soup, imdb_id)
            
            if not new_reviews:  # 더 이상 리뷰 없음
                break
            
            all_reviews.extend(new_reviews)
            page_count += 1
            
            # 다음 키 찾기
            load_more = ajax_soup.select_one("div.load-more-data")
            pagination_key = load_more.get("data-key") if load_more else None
            
            # 약간의 딜레이 (rate limit 방지)
            await asyncio.sleep(0.5)
        
        stats["reviews_total"] += len(all_reviews)
        return all_reviews
    
    except Exception as e:
        print(f"❌ Error for {imdb_id} ({series_title}): {e}")
        return all_reviews

# ==========================================================
# 체크포인트 관리
# ==========================================================
def save_checkpoint(processed_ids, results):
    """중간 저장"""
    checkpoint = {
        'processed_ids': list(processed_ids),
        'stats': stats.copy(),
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)
    
    # 중간 결과도 저장
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

def load_checkpoint():
    """체크포인트 로드"""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            return set(checkpoint.get('processed_ids', []))
    return set()

# ==========================================================
# 메인 실행
# ==========================================================
async def main(input_csv_path, vote_threshold=10, max_reviews_per_page=None):
    """
    전체 리뷰 수집
    
    Args:
        input_csv_path: TMDB CSV 파일
        vote_threshold: 최소 vote_count
        max_reviews_per_page: 페이지당 최대 리뷰 수 (None이면 전체)
    """
    print("=" * 90)
    print("🎬 IMDB 전체 리뷰 크롤러 (비동기 최적화)")
    print("=" * 90)
    
    stats["start_time"] = datetime.now()
    t0 = datetime.now()
    
    # 1. 데이터 로드
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv(input_csv_path)
    df_filtered = df[(df['vote_count'] >= vote_threshold) & (df['imdb_id'].notna())]
    
    print(f"✅ 전체 시리즈: {len(df):,}개")
    print(f"✅ 필터링 (vote_count>={vote_threshold} & imdb_id 존재): {len(df_filtered):,}개")
    
    if len(df_filtered) == 0:
        print("⚠️  조건을 만족하는 데이터가 없습니다.")
        return
    
    # 2. 체크포인트 로드
    processed_ids = load_checkpoint()
    series_list = df_filtered[['id', 'title', 'imdb_id']].to_dict('records')
    
    if processed_ids:
        print(f"📌 체크포인트 로드: {len(processed_ids):,}개 처리 완료")
        series_list = [s for s in series_list if s['imdb_id'] not in processed_ids]
        print(f"📌 남은 작업: {len(series_list):,}개")
    
    if len(series_list) == 0:
        print("✅ 모든 데이터가 이미 처리되었습니다.")
        return
    
    stats["series_total"] = len(series_list)
    
    # 3. 크롤링
    print(f"\n🚀 크롤링 시작")
    print(f"⚙️  Rate Limit: {MAX_CALLS_PER_SECOND}회/초")
    print(f"⏱️  예상 시간: 시리즈당 평균 30초 → 총 {len(series_list)*30/60:.0f}분")
    
    connector = aiohttp.TCPConnector(
        limit=10,
        force_close=False,
        enable_cleanup_closed=True
    )
    
    all_results = []
    batch_size = 10  # 한 번에 처리할 시리즈 수
    
    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        for i in range(0, len(series_list), batch_size):
            batch = series_list[i:i+batch_size]
            
            # 배치 처리
            tasks = [
                fetch_all_reviews_for_series(session, s['imdb_id'], s['title'], max_reviews_per_page)
                for s in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 수집
            for series, reviews in zip(batch, batch_results):
                if isinstance(reviews, list):
                    all_results.extend(reviews)
                    processed_ids.add(series['imdb_id'])
                    stats["series_success"] += 1
                    print(f"✅ {series['title']}: {len(reviews):,}개 리뷰")
                else:
                    stats["series_failed"] += 1
                    print(f"❌ {series['title']}: 실패")
            
            # 주기적 저장
            if (i + batch_size) % 50 == 0:
                save_checkpoint(processed_ids, all_results)
            
            # 진행 상황
            elapsed = (datetime.now() - t0).total_seconds() / 60
            progress = stats["series_success"] + stats["series_failed"]
            rate = progress / elapsed if elapsed > 0 else 0
            eta = (stats["series_total"] - progress) / rate if rate > 0 else 0
            
            print(f"\n📊 진행: {progress}/{stats['series_total']} ({progress/stats['series_total']*100:.1f}%) | "
                  f"성공: {stats['series_success']} | 실패: {stats['series_failed']} | "
                  f"총 리뷰: {stats['reviews_total']:,}개 | "
                  f"요청: {stats['requests']:,}회 | "
                  f"ETA: {eta:.0f}분\n")
    
    # 4. 최종 저장
    print("\n💾 최종 저장 중...")
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    try:
        df_results.to_parquet(OUTPUT_PARQUET, index=False)
    except:
        pass
    
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
    
    # 5. 통계
    elapsed = (datetime.now() - t0).total_seconds() / 60
    
    print("\n" + "=" * 90)
    print("🎉 크롤링 완료!")
    print("=" * 90)
    print(f"📌 시리즈: {stats['series_success']:,}/{stats['series_total']:,}개 성공")
    print(f"📌 총 리뷰: {len(df_results):,}개")
    print(f"📌 평균: {len(df_results)/stats['series_success']:.1f}개/시리즈")
    print(f"📌 총 요청: {stats['requests']:,}회")
    print(f"⏱️  총 시간: {elapsed:.1f}분 ({elapsed/60:.2f}시간)")
    print(f"📊 속도: {stats['series_success']/elapsed:.1f}개/분")
    print("=" * 90)
    
    # 샘플
    print("\n📊 샘플 데이터:")
    print(df_results.head(3).to_string())
    print(f"\n✅ 결과 파일: {OUTPUT_CSV}")

# ==========================================================
# 실행
# ==========================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IMDB 전체 리뷰 크롤러')
    parser.add_argument('--input', '-i', default='tv_series_2013_0101_0215_FULL.csv',
                        help='입력 CSV 파일')
    parser.add_argument('--vote', '-v', type=int, default=10,
                        help='최소 vote_count (기본: 10)')
    parser.add_argument('--max-pages', '-m', type=int, default=None,
                        help='시리즈당 최대 페이지 수 (기본: 전체)')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {args.input}")
    else:
        asyncio.run(main(args.input, args.vote, args.max_pages))
