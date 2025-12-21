"""
TMDB TV Series Data Collector - Async Version
비동기 방식으로 TMDB API를 통해 TV 시리즈 데이터 수집
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import List, Dict, Optional
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TMDBAsyncCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {"accept": "application/json"}
        
        # Rate limiting: 40 req/s
        self.rate_limit = 40
        self.rate_period = 1.0
        self.semaphore = asyncio.Semaphore(self.rate_limit)
        
        # 진행 상황 저장
        self.checkpoint_file = "checkpoint_async.json"
        self.output_file = "tmdb_tv_series_data.parquet"
        
        # 데이터 저장
        self.collected_data = []
        
    async def _rate_limited_request(self, session: aiohttp.ClientSession, url: str, params: dict) -> Optional[dict]:
        """Rate limiting을 적용한 API 요청"""
        async with self.semaphore:
            try:
                async with session.get(url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Too Many Requests
                        retry_after = int(response.headers.get('Retry-After', 1))
                        logger.warning(f"Rate limited. Waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._rate_limited_request(session, url, params)
                    else:
                        logger.error(f"Error {response.status}: {url}")
                        return None
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return None
            finally:
                await asyncio.sleep(1.0 / self.rate_limit)
    
    async def fetch_discover_page(self, session: aiohttp.ClientSession, 
                                  start_date: str, end_date: str, page: int) -> Optional[dict]:
        """Discover API로 TV 시리즈 목록 가져오기"""
        url = f"{self.base_url}/discover/tv"
        params = {
            "api_key": self.api_key,
            "first_air_date.gte": start_date,
            "first_air_date.lte": end_date,
            "vote_count.gte": 30,
            "include_adult": "false",
            "include_null_first_air_dates": "false",
            "language": "en-US",
            "page": page,
            "sort_by": "first_air_date.asc"
        }
        return await self._rate_limited_request(session, url, params)
    
    async def fetch_external_ids(self, session: aiohttp.ClientSession, series_id: int) -> Optional[dict]:
        """External IDs (IMDB ID 포함) 가져오기"""
        url = f"{self.base_url}/tv/{series_id}/external_ids"
        params = {"api_key": self.api_key}
        return await self._rate_limited_request(session, url, params)
    
    async def fetch_credits(self, session: aiohttp.ClientSession, series_id: int) -> Optional[dict]:
        """크레딧 정보 (배우, 제작진) 가져오기"""
        url = f"{self.base_url}/tv/{series_id}/credits"
        params = {
            "api_key": self.api_key,
            "language": "en-US"
        }
        return await self._rate_limited_request(session, url, params)
    
    def extract_executive_producers(self, crew_list: List[dict]) -> Dict[str, str]:
        """총괄 프로듀서 정보 추출"""
        producers = [c for c in crew_list if c.get("job") == "Executive Producer"]
        
        if not producers:
            return {
                "executive_producer_name": "",
                "executive_producer_ids": "",
                "executive_producer_gender": "",
                "executive_producer_profile_path": ""
            }
        
        return {
            "executive_producer_name": "; ".join([p.get("name", "") for p in producers]),
            "executive_producer_ids": "; ".join([str(p.get("id", "")) for p in producers]),
            "executive_producer_gender": "; ".join([str(p.get("gender", "")) for p in producers]),
            "executive_producer_profile_path": "; ".join([p.get("profile_path", "") or "" for p in producers])
        }
    
    def extract_writers(self, crew_list: List[dict]) -> Dict[str, str]:
        """작가 정보 추출"""
        writer_jobs = {"Writer", "Screenplay", "Story", "Creator", "Novel", "Comic Book"}
        writers = [
            c for c in crew_list
            if c.get("job") in writer_jobs or c.get("department") == "Writing"
        ]
        
        if not writers:
            return {
                "writers_name": "",
                "writer_roles": "",
                "writer_ids": "",
                "writer_gender": "",
                "writer_profile_path": ""
            }
        
        return {
            "writers_name": "; ".join([w.get("name", "") for w in writers]),
            "writer_roles": "; ".join([w.get("job", "") for w in writers]),
            "writer_ids": "; ".join([str(w.get("id", "")) for w in writers]),
            "writer_gender": "; ".join([str(w.get("gender", "")) for w in writers]),
            "writer_profile_path": "; ".join([w.get("profile_path", "") or "" for w in writers])
        }
    
    def extract_top_cast(self, cast_list: List[dict]) -> Dict[str, str]:
        """주연 배우 상위 5명 정보 추출 (order 기준)"""
        # order 필드로 정렬하고 상위 5명만
        sorted_cast = sorted(cast_list, key=lambda x: x.get("order", 999))
        top_cast = [c for c in sorted_cast if c.get("character")][:5]
        
        if not top_cast:
            return {
                "top_cast_order": "",
                "top_cast": "",
                "character": "",
                "top_cast_ids": "",
                "top_cast_gender": "",
                "top_cast_profile_path": ""
            }
        
        return {
            "top_cast_order": "; ".join([str(c.get("order", "")) for c in top_cast]),
            "top_cast": "; ".join([c.get("name", "") for c in top_cast]),
            "character": "; ".join([c.get("character", "") for c in top_cast]),
            "top_cast_ids": "; ".join([str(c.get("id", "")) for c in top_cast]),
            "top_cast_gender": "; ".join([str(c.get("gender", "")) for c in top_cast]),
            "top_cast_profile_path": "; ".join([c.get("profile_path", "") or "" for c in top_cast])
        }
    
    async def process_series(self, session: aiohttp.ClientSession, series_data: dict) -> Optional[dict]:
        """개별 TV 시리즈 처리"""
        series_id = series_data.get("id")
        
        # External IDs 가져오기
        external_ids = await self.fetch_external_ids(session, series_id)
        if not external_ids or not external_ids.get("imdb_id"):
            logger.debug(f"Series {series_id} has no IMDB ID, skipping")
            return None
        
        # Credits 가져오기
        credits = await self.fetch_credits(session, series_id)
        if not credits:
            logger.debug(f"Failed to fetch credits for series {series_id}")
            return None
        
        # 데이터 추출
        cast_list = credits.get("cast", [])
        crew_list = credits.get("crew", [])
        
        row_data = {
            "imdb_id": external_ids.get("imdb_id"),
            "series_id": series_id,
            "title": series_data.get("name", ""),
            "original_name": series_data.get("original_name", ""),
            "first_air_date": series_data.get("first_air_date", ""),
            "vote_average": series_data.get("vote_average", 0),
            "vote_count": series_data.get("vote_count", 0),
            "popularity": series_data.get("popularity", 0),
        }
        
        # 총괄 프로듀서, 작가, 주연 배우 정보 추가
        row_data.update(self.extract_executive_producers(crew_list))
        row_data.update(self.extract_writers(crew_list))
        row_data.update(self.extract_top_cast(cast_list))
        
        return row_data
    
    async def collect_date_range(self, session: aiohttp.ClientSession, 
                                start_date: str, end_date: str) -> None:
        """특정 날짜 범위의 데이터 수집"""
        logger.info(f"Collecting: {start_date} to {end_date}")
        page = 1
        
        while page <= 500:  # TMDB 500페이지 제한
            result = await self.fetch_discover_page(session, start_date, end_date, page)
            
            if not result or "results" not in result:
                break
            
            results = result["results"]
            if not results:
                break
            
            total_pages = min(result.get("total_pages", 0), 500)
            logger.info(f"Processing page {page}/{total_pages} ({start_date} ~ {end_date})")
            
            # 각 시리즈 처리
            tasks = [self.process_series(session, series) for series in results]
            processed_data = await asyncio.gather(*tasks)
            
            # 유효한 데이터만 추가
            valid_data = [d for d in processed_data if d is not None]
            self.collected_data.extend(valid_data)
            
            logger.info(f"Collected {len(valid_data)} valid series from this page")
            
            if page >= total_pages:
                break
            
            page += 1
        
        logger.info(f"Completed range {start_date} ~ {end_date}. Total collected: {len(self.collected_data)}")
    
    def generate_date_ranges(self, start_date: str, end_date: str, months: int = 3) -> List[tuple]:
        """날짜 범위를 여러 구간으로 분할 (500페이지 제한 대응)"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        ranges = []
        current = start
        
        while current < end:
            next_date = current + timedelta(days=months * 30)
            if next_date > end:
                next_date = end
            
            ranges.append((
                current.strftime("%Y-%m-%d"),
                next_date.strftime("%Y-%m-%d")
            ))
            current = next_date + timedelta(days=1)
        
        return ranges
    
    def save_checkpoint(self, completed_ranges: List[tuple]) -> None:
        """진행 상황 저장"""
        checkpoint = {
            "completed_ranges": completed_ranges,
            "collected_count": len(self.collected_data),
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved: {len(completed_ranges)} ranges completed")
    
    def load_checkpoint(self) -> List[tuple]:
        """저장된 진행 상황 불러오기"""
        if not Path(self.checkpoint_file).exists():
            return []
        
        with open(self.checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint: {checkpoint['collected_count']} series previously collected")
            return [tuple(r) for r in checkpoint.get("completed_ranges", [])]
    
    def save_to_parquet(self) -> None:
        """수집한 데이터를 Parquet 파일로 저장"""
        if not self.collected_data:
            logger.warning("No data to save")
            return
        
        df = pd.DataFrame(self.collected_data)
        df.to_parquet(self.output_file, index=False, engine='pyarrow')
        logger.info(f"Data saved to {self.output_file}: {len(df)} series")
    
    async def run(self, start_date: str = "2005-01-01", end_date: str = "2025-11-30") -> None:
        """메인 수집 프로세스"""
        logger.info("Starting TMDB TV Series collection (Async)")
        logger.info(f"Period: {start_date} ~ {end_date}")
        
        # 날짜 범위 생성
        all_ranges = self.generate_date_ranges(start_date, end_date, months=3)
        completed_ranges = self.load_checkpoint()
        
        # 아직 수집하지 않은 범위 필터링
        remaining_ranges = [r for r in all_ranges if r not in completed_ranges]
        logger.info(f"Total ranges: {len(all_ranges)}, Remaining: {len(remaining_ranges)}")
        
        async with aiohttp.ClientSession() as session:
            for i, (range_start, range_end) in enumerate(remaining_ranges, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Range {i}/{len(remaining_ranges)}")
                
                await self.collect_date_range(session, range_start, range_end)
                
                # 체크포인트 저장
                completed_ranges.append((range_start, range_end))
                self.save_checkpoint(completed_ranges)
                
                # 중간 저장 (매 5개 범위마다)
                if i % 5 == 0:
                    self.save_to_parquet()
        
        # 최종 저장
        self.save_to_parquet()
        logger.info(f"\n{'='*60}")
        logger.info(f"Collection completed! Total series: {len(self.collected_data)}")


async def main():
    # TODO: TMDB API 키를 여기에 입력하세요
    API_KEY = "YOUR_TMDB_API_KEY_HERE"
    
    if API_KEY == "YOUR_TMDB_API_KEY_HERE":
        print("Error: Please set your TMDB API key in the script")
        return
    
    collector = TMDBAsyncCollector(API_KEY)
    await collector.run(start_date="2005-01-01", end_date="2025-11-30")


if __name__ == "__main__":
    asyncio.run(main())
