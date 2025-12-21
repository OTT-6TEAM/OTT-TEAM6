# ==========================================================
# IMDB 전체 리뷰 크롤러 - 개선된 동기 버전
# 사용자 제공 코드 기반 + 에러 처리 + 체크포인트 추가
# ==========================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import json
from pathlib import Path
from datetime import datetime

# ==========================================================
# 설정
# ==========================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

SLEEP_BETWEEN_SERIES = 1.5  # 시리즈 간 대기 시간
SLEEP_BETWEEN_PAGES = 0.5   # 페이지 간 대기 시간
MAX_RETRIES = 3              # 재시도 횟수

OUTPUT_CSV = "imdb_reviews_full_sync.csv"
CHECKPOINT_FILE = "imdb_checkpoint_sync.json"
FAILED_FILE = "imdb_failed_ids.txt"

# 통계
stats = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "reviews": 0,
    "start_time": None
}

# ==========================================================
# 리뷰 파싱
# ==========================================================

def parse_review_block(soup, imdb_id):
    """
    soup에서 리뷰 리스트 파싱 (개선)
    """
    review_blocks = soup.select(".review-container")
    reviews = []
    
    for block in review_blocks:
        # 제목
        title = block.select_one(".title")
        
        # 내용
        content = block.select_one(".text.show-more__control")
        if not content:
            content = block.select_one(".text")
        
        # 평점
        rating = block.select_one(".rating-other-user-rating span")
        
        # 작성자
        author = block.select_one(".display-name-link a")
        if not author:
            author = block.select_one(".display-name-link")
        
        # 날짜
        date = block.select_one(".review-date")
        
        # Helpful 투표
        helpful = None
        actions = block.select_one(".actions")
        if actions:
            import re
            match = re.search(r'(\d+)\s+out of\s+(\d+)', actions.get_text())
            if match:
                helpful = f"{match.group(1)}/{match.group(2)}"
        
        # Spoiler 여부
        spoiler = "spoiler-warning" in str(block)
        
        reviews.append({
            "imdb_id": imdb_id,
            "review_title": title.get_text(strip=True) if title else None,
            "review_text": content.get_text(strip=True) if content else None,
            "review_rating": rating.get_text(strip=True) if rating else None,
            "review_author": author.get_text(strip=True) if author else None,
            "review_date": date.get_text(strip=True) if date else None,
            "helpful_votes": helpful,
            "is_spoiler": spoiler,
        })
    
    return reviews

# ==========================================================
# 전체 리뷰 수집 (재시도 로직 추가)
# ==========================================================

def fetch_all_imdb_reviews(imdb_id, series_title="", max_pages=None):
    """
    IMDb 전체 리뷰 크롤링 (paginationKey 이용)
    
    Args:
        imdb_id: IMDB ID
        series_title: 시리즈 제목 (로깅용)
        max_pages: 최대 페이지 수 (None이면 전체)
    
    Returns:
        list: 리뷰 리스트
    """
    base_url = f"https://www.imdb.com/title/{imdb_id}/reviews"
    ajax_url = f"https://www.imdb.com/title/{imdb_id}/reviews/_ajax"
    all_reviews = []
    page_count = 0
    
    try:
        # 1. 첫 페이지 요청 (재시도 포함)
        for attempt in range(MAX_RETRIES):
            try:
                res = requests.get(base_url, headers=HEADERS, timeout=15)
                res.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # 지수 백오프
        
        soup = BeautifulSoup(res.text, "html.parser")
        
        # 첫 리뷰 파싱
        all_reviews.extend(parse_review_block(soup, imdb_id))
        page_count += 1
        
        # 첫 pagination key
        load_more = soup.select_one("div.load-more-data")
        if load_more is None:
            return all_reviews
        
        pagination_key = load_more.get("data-key")
        
        # 2. Ajax 요청 반복
        while pagination_key:
            if max_pages and page_count >= max_pages:
                break
            
            # POST 요청 (재시도 포함)
            for attempt in range(MAX_RETRIES):
                try:
                    payload = {"paginationKey": pagination_key}
                    res = requests.post(ajax_url, headers=HEADERS, data=payload, timeout=15)
                    res.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"⚠️  {series_title}: 페이지 {page_count+1} 실패")
                        return all_reviews
                    time.sleep(2 ** attempt)
            
            ajax_soup = BeautifulSoup(res.text, "html.parser")
            
            # 리뷰 추가
            new_reviews = parse_review_block(ajax_soup, imdb_id)
            if not new_reviews:  # 더 이상 없으면 종료
                break
            
            all_reviews.extend(new_reviews)
            page_count += 1
            
            # 다음 키 탐색
            load_more = ajax_soup.select_one("div.load-more-data")
            pagination_key = load_more.get("data-key") if load_more else None
            
            time.sleep(SLEEP_BETWEEN_PAGES)  # IMDb block 방지
        
        return all_reviews
    
    except Exception as e:
        print(f"❌ {series_title} ({imdb_id}): {str(e)[:100]}")
        return all_reviews

# ==========================================================
# 체크포인트 관리
# ==========================================================

def save_checkpoint(processed_ids, failed_ids):
    """진행 상황 저장"""
    checkpoint = {
        'processed_ids': list(processed_ids),
        'failed_ids': list(failed_ids),
        'stats': stats.copy(),
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    # 실패 목록 별도 저장
    if failed_ids:
        with open(FAILED_FILE, 'w') as f:
            f.write('\n'.join(failed_ids))

def load_checkpoint():
    """체크포인트 로드"""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            return (
                set(checkpoint.get('processed_ids', [])),
                set(checkpoint.get('failed_ids', []))
            )
    return set(), set()

# ==========================================================
# 메인 실행 함수
# ==========================================================

def collect_all_reviews(input_csv, vote_threshold=10, max_pages=None, save_interval=20):
    """
    전체 TV 시리즈 리뷰 수집
    
    Args:
        input_csv: TMDB CSV 파일 경로
        vote_threshold: 최소 vote_count
        max_pages: 시리즈당 최대 페이지 수
        save_interval: 중간 저장 간격
    """
    print("=" * 90)
    print("🎬 IMDB 전체 리뷰 크롤러 (개선된 동기 버전)")
    print("=" * 90)
    
    stats["start_time"] = datetime.now()
    t0 = datetime.now()
    
    # 1. 데이터 로드
    print("\n📂 데이터 로드 중...")
    df_series = pd.read_csv(input_csv)
    df_target = df_series[
        (df_series['vote_count'] >= vote_threshold) & 
        (df_series['imdb_id'].notna())
    ]
    
    print(f"✅ 전체 시리즈: {len(df_series):,}개")
    print(f"✅ 필터링 (vote_count>={vote_threshold} & imdb_id 존재): {len(df_target):,}개")
    
    if len(df_target) == 0:
        print("⚠️  조건을 만족하는 데이터가 없습니다.")
        return
    
    # 2. 체크포인트 로드
    processed_ids, failed_ids = load_checkpoint()
    
    if processed_ids:
        print(f"📌 체크포인트 로드: {len(processed_ids):,}개 처리 완료, {len(failed_ids)}개 실패")
        df_target = df_target[~df_target['imdb_id'].isin(processed_ids)]
        print(f"📌 남은 작업: {len(df_target):,}개")
    
    if len(df_target) == 0:
        print("✅ 모든 데이터가 이미 처리되었습니다.")
        return
    
    stats["total"] = len(df_target)
    
    # 3. 크롤링
    print(f"\n🚀 크롤링 시작")
    print(f"⚙️  대기 시간: 시리즈 간 {SLEEP_BETWEEN_SERIES}초, 페이지 간 {SLEEP_BETWEEN_PAGES}초")
    print(f"⏱️  예상 시간: {len(df_target) * SLEEP_BETWEEN_SERIES / 60:.0f}분 (최소)")
    
    all_reviews = []
    
    # 기존 데이터 로드 (이어서 저장하기 위해)
    if Path(OUTPUT_CSV).exists():
        existing_df = pd.read_csv(OUTPUT_CSV)
        all_reviews = existing_df.to_dict('records')
        print(f"📌 기존 데이터 로드: {len(all_reviews):,}개 리뷰")
    
    # tqdm으로 진행 상황 표시
    for idx, row in tqdm(df_target.iterrows(), total=len(df_target), desc="수집 중"):
        imdb_id = row['imdb_id']
        title = row.get('title', 'Unknown')
        
        try:
            reviews = fetch_all_imdb_reviews(imdb_id, title, max_pages)
            
            if reviews:
                all_reviews.extend(reviews)
                processed_ids.add(imdb_id)
                stats["success"] += 1
                stats["reviews"] += len(reviews)
                tqdm.write(f"✅ {title}: {len(reviews):,}개 리뷰")
            else:
                processed_ids.add(imdb_id)
                failed_ids.add(imdb_id)
                stats["failed"] += 1
                tqdm.write(f"⚠️  {title}: 리뷰 없음")
        
        except Exception as e:
            failed_ids.add(imdb_id)
            stats["failed"] += 1
            tqdm.write(f"❌ {title}: {str(e)[:50]}")
        
        # 주기적 저장
        if (stats["success"] + stats["failed"]) % save_interval == 0:
            df_temp = pd.DataFrame(all_reviews)
            df_temp.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            save_checkpoint(processed_ids, failed_ids)
            
            # 진행 상황
            elapsed = (datetime.now() - t0).total_seconds() / 60
            progress = stats["success"] + stats["failed"]
            rate = progress / elapsed if elapsed > 0 else 0
            eta = (stats["total"] - progress) / rate if rate > 0 else 0
            
            tqdm.write(f"\n📊 {progress}/{stats['total']} ({progress/stats['total']*100:.1f}%) | "
                      f"성공: {stats['success']} | 실패: {stats['failed']} | "
                      f"리뷰: {stats['reviews']:,}개 | ETA: {eta:.0f}분\n")
        
        time.sleep(SLEEP_BETWEEN_SERIES)
    
    # 4. 최종 저장
    print("\n💾 최종 저장 중...")
    df_reviews = pd.DataFrame(all_reviews)
    df_reviews.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    try:
        df_reviews.to_parquet(OUTPUT_CSV.replace('.csv', '.parquet'), index=False)
    except:
        pass
    
    # 체크포인트 삭제
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
    
    # 5. 통계
    elapsed = (datetime.now() - t0).total_seconds() / 60
    
    print("\n" + "=" * 90)
    print("🎉 크롤링 완료!")
    print("=" * 90)
    print(f"📌 처리: {stats['success'] + stats['failed']}/{stats['total']}개")
    print(f"📌 성공: {stats['success']}개 ({stats['success']/(stats['success']+stats['failed'])*100:.1f}%)")
    print(f"📌 실패: {stats['failed']}개")
    print(f"📌 총 리뷰: {len(df_reviews):,}개")
    print(f"📌 평균: {len(df_reviews)/stats['success']:.1f}개/시리즈")
    print(f"⏱️  총 시간: {elapsed:.1f}분 ({elapsed/60:.2f}시간)")
    print(f"📊 속도: {stats['success']/elapsed:.1f}개/분")
    print("=" * 90)
    
    # 샘플
    print("\n📊 샘플 데이터:")
    print(df_reviews.head(3))
    print(f"\n✅ 결과 파일: {OUTPUT_CSV}")
    
    if failed_ids:
        print(f"⚠️  실패 목록: {FAILED_FILE}")

# ==========================================================
# 실행
# ==========================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IMDB 리뷰 크롤러 (동기)')
    parser.add_argument('--input', '-i', default='tv_series_2013_0101_0215_FULL.csv',
                        help='입력 CSV 파일')
    parser.add_argument('--vote', '-v', type=int, default=10,
                        help='최소 vote_count')
    parser.add_argument('--max-pages', '-m', type=int, default=None,
                        help='시리즈당 최대 페이지 수')
    parser.add_argument('--save-interval', '-s', type=int, default=20,
                        help='중간 저장 간격')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {args.input}")
    else:
        collect_all_reviews(
            args.input,
            args.vote,
            args.max_pages,
            args.save_interval
        )
