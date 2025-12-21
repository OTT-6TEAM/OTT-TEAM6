# IMDB 데이터 크롤링 가이드

## 필요한 라이브러리 설치

```bash
pip install pandas requests beautifulsoup4 tqdm --break-system-packages
```

## 사용 방법

1. CSV 파일(`tv_series_2005_2015_FULL.csv`)을 스크립트와 같은 폴더에 준비
2. 스크립트 실행:
   ```bash
   python imdb_scraper.py
   ```

## 출력 파일

- `tv_series_with_ratings.csv`: 원본 데이터 + IMDB rating/Metascore
- `imdb_ratings_only.csv`: IMDB ID, rating, Metascore만 포함

## 주요 기능

1. **필터링**: vote_count >= 30이고 imdb_id가 있는 데이터만 크롤링
2. **Rate Limiting**: 각 요청 사이에 1초 딜레이 (IMDB 서버 보호)
3. **에러 처리**: 크롤링 실패 시에도 계속 진행
4. **진행상황**: tqdm을 통한 진행률 표시

## 주의사항

⚠️ **중요**:
- IMDB 크롤링은 많은 시간이 소요될 수 있습니다 (시리즈 수 × 1초)
- IMDB의 구조 변경으로 선택자가 작동하지 않을 수 있습니다
- 너무 많은 요청은 IP 차단으로 이어질 수 있으니 주의하세요
- IMDB의 이용 약관을 확인하고 준수하세요

## 컬럼 설명

- `imdb_id`: IMDB 고유 ID (예: tt26443597)
- `imdb_rating`: IMDB 평점 (0-10)
- `metascore`: Metacritic 점수 (0-100)
- `url`: IMDB 페이지 URL
- `status`: 크롤링 상태 ('success' 또는 에러 메시지)

## 문제 해결

### Rating/Metascore가 None으로 나오는 경우
- IMDB 페이지 구조가 변경되었을 수 있습니다
- 해당 시리즈에 실제로 점수가 없을 수 있습니다
- 브라우저에서 직접 URL을 확인해보세요

### 너무 느린 경우
- `time.sleep(1)`의 값을 줄일 수 있지만, IP 차단 위험이 있습니다
- 멀티스레딩을 사용할 수 있지만, 이 역시 주의가 필요합니다

### IP 차단된 경우
- 시간을 두고 다시 시도하세요
- VPN 사용을 고려하세요
- delay 시간을 늘리세요
