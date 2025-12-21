# ==========================================================
# IMDB GraphQL API 크롤러 - 테스트 스크립트
# 소량의 데이터로 빠르게 테스트
# ==========================================================

import asyncio
import aiohttp
import json
from imdb_graphql_crawler import (
    build_graphql_url,
    fetch_graphql,
    parse_review_node,
    fetch_all_reviews_for_series,
    RateLimiter,
    rate_limiter
)

# 테스트용 IMDB ID
TEST_SERIES = [
    {"imdb_id": "tt0944947", "title": "Game of Thrones"},
    {"imdb_id": "tt0903747", "title": "Breaking Bad"},
    {"imdb_id": "tt2306299", "title": "Vikings"},
]

async def test_graphql_api():
    """GraphQL API 기본 테스트"""
    print("=" * 70)
    print("🧪 IMDB GraphQL API 테스트")
    print("=" * 70)
    
    connector = aiohttp.TCPConnector(limit=5)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for series in TEST_SERIES:
            imdb_id = series['imdb_id']
            title = series['title']
            
            print(f"\n📥 테스트 중: {title} ({imdb_id})")
            
            # 첫 페이지만 가져오기
            url = build_graphql_url(imdb_id, first=5)
            print(f"🔗 URL (첫 5개 리뷰): {url[:100]}...")
            
            response = await fetch_graphql(session, url)
            
            if response:
                # 구조 확인
                data = response.get('data', {})
                title_data = data.get('title', {})
                reviews_data = title_data.get('reviews', {})
                
                total = reviews_data.get('total', 0)
                edges = reviews_data.get('edges', [])
                
                print(f"   ✅ 총 리뷰 수: {total:,}개")
                print(f"   ✅ 받은 리뷰: {len(edges)}개")
                
                # 첫 번째 리뷰 파싱
                if edges:
                    first_review = parse_review_node(edges[0]['node'], imdb_id)
                    if first_review:
                        print(f"\n   📝 첫 번째 리뷰:")
                        print(f"      작성자: {first_review['username']}")
                        print(f"      평점: {first_review['author_rating']}/10")
                        print(f"      날짜: {first_review['submission_date']}")
                        print(f"      제목: {first_review['review_title']}")
                        print(f"      내용: {first_review['review_text'][:100]}...")
                        print(f"      Helpful: {first_review['helpful_up_votes']}/{first_review['helpful_total']}")
            else:
                print(f"   ❌ API 호출 실패")
            
            await asyncio.sleep(1)
    
    print("\n" + "=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)

async def test_full_collection():
    """전체 리뷰 수집 테스트 (1개 시리즈만)"""
    print("\n" + "=" * 70)
    print("🧪 전체 리뷰 수집 테스트 (Game of Thrones)")
    print("=" * 70)
    
    connector = aiohttp.TCPConnector(limit=5)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Game of Thrones 리뷰 100개만 수집
        reviews = await fetch_all_reviews_for_series(
            session,
            "tt0944947",
            "Game of Thrones",
            max_reviews=100
        )
        
        print(f"\n✅ 수집된 리뷰: {len(reviews)}개")
        
        if reviews:
            import pandas as pd
            df = pd.DataFrame(reviews)
            
            print("\n📊 통계:")
            print(f"   평균 평점: {df['author_rating'].mean():.2f}/10")
            print(f"   평균 길이: {df['review_text_length'].mean():.0f}자")
            print(f"   Spoiler: {df['is_spoiler'].sum()}개")
            
            # 샘플 저장
            df.to_csv('test_reviews_sample.csv', index=False, encoding='utf-8-sig')
            print(f"\n✅ 샘플 저장: test_reviews_sample.csv")

async def test_url_generation():
    """URL 생성 테스트"""
    print("\n" + "=" * 70)
    print("🧪 URL 생성 테스트")
    print("=" * 70)
    
    # 첫 페이지
    url1 = build_graphql_url("tt0944947", first=25)
    print(f"\n1️⃣ 첫 페이지 URL:")
    print(f"   {url1[:150]}...")
    
    # 두 번째 페이지 (커서 있음)
    url2 = build_graphql_url(
        "tt0944947",
        after_cursor="g4xopermtizcsyya76whvnburdr4yazs3modv7pjdpj3qflanarkwdc6oi2u7w5il4pln667fmielj3jr4cuobss",
        first=25
    )
    print(f"\n2️⃣ 두 번째 페이지 URL (커서 포함):")
    print(f"   {url2[:150]}...")
    
    # 정렬 기준 변경
    url3 = build_graphql_url("tt0944947", first=25, sort_by="SUBMISSION_DATE")
    print(f"\n3️⃣ 날짜 정렬 URL:")
    print(f"   {url3[:150]}...")

if __name__ == "__main__":
    import sys
    
    print("""
╔═══════════════════════════════════════════════════════╗
║     IMDB GraphQL API 크롤러 - 테스트 메뉴            ║
╚═══════════════════════════════════════════════════════╝

1. URL 생성 테스트 (빠름)
2. API 호출 테스트 (3개 시리즈)
3. 전체 수집 테스트 (1개 시리즈, 100개 리뷰)
4. 모두 실행

선택: """, end='')
    
    try:
        choice = input().strip()
    except:
        choice = "4"
    
    if choice == "1":
        asyncio.run(test_url_generation())
    elif choice == "2":
        asyncio.run(test_graphql_api())
    elif choice == "3":
        asyncio.run(test_full_collection())
    else:
        asyncio.run(test_url_generation())
        asyncio.run(test_graphql_api())
        asyncio.run(test_full_collection())
    
    print("\n✨ 테스트가 성공했다면 본격적인 크롤링을 시작하세요:")
    print("   python imdb_graphql_crawler.py --vote 30")
