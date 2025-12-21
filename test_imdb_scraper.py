# ==========================================================
# IMDB 크롤러 테스트 스크립트
# 소량의 데이터로 빠르게 테스트
# ==========================================================

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import json

# 테스트용 IMDB ID들 (유명한 TV 시리즈)
TEST_DATA = [
    {"imdb_id": "tt0944947", "title": "Game of Thrones"},
    {"imdb_id": "tt0903747", "title": "Breaking Bad"},
    {"imdb_id": "tt2306299", "title": "The Vikings"},
]

async def quick_test():
    """빠른 테스트 - 3개 시리즈만"""
    print("🧪 IMDB 크롤러 테스트 시작\n")
    
    # imdb_scraper 모듈 import
    try:
        # 현재 디렉토리의 imdb_scraper를 import
        import sys
        sys.path.insert(0, '/home/claude')
        from imdb_scraper import scrape_imdb_data
    except ImportError:
        print("❌ imdb_scraper.py 파일이 필요합니다.")
        return
    
    connector = aiohttp.TCPConnector(limit=5)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        results = []
        
        for item in TEST_DATA:
            print(f"📥 수집 중: {item['title']} ({item['imdb_id']})")
            result = await scrape_imdb_data(session, item['imdb_id'], item['title'])
            results.append(result)
            
            # 결과 출력
            if result['imdb_rating']:
                print(f"   ⭐ 평점: {result['imdb_rating']}/10 ({result['imdb_rating_count']:,}표)")
            if result['meta_score']:
                print(f"   🎯 메타스코어: {result['meta_score']}/100")
            if result['reviews_json']:
                reviews = json.loads(result['reviews_json'])
                print(f"   💬 리뷰: {len(reviews)}개 수집")
            print()
            
            # 짧은 대기 (너무 빠르게 요청하지 않도록)
            await asyncio.sleep(1)
    
    # 결과 저장
    df = pd.DataFrame(results)
    df.to_csv('imdb_test_results.csv', index=False, encoding='utf-8-sig')
    
    print("✅ 테스트 완료!")
    print(f"📁 결과 파일: imdb_test_results.csv")
    print("\n" + "="*60)
    print("테스트가 성공했다면 본격적인 크롤링을 시작하세요:")
    print("python imdb_scraper.py")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(quick_test())
