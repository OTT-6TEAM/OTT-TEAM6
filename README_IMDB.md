# IMDB 데이터 크롤러 🎬

TMDB TV 시리즈 데이터에서 IMDB 정보를 추가로 수집하는 비동기 크롤러입니다.

## 🚀 빠른 시작

### 1. 패키지 설치
```bash
pip install aiohttp beautifulsoup4 pandas --break-system-packages
```

### 2. 실행
```bash
python imdb_scraper.py
```

## 📊 수집 데이터

- ⭐ **IMDB Rating**: 평점 (0-10점)
- 👥 **Rating Count**: 평점 투표 수
- 🎯 **Meta Score**: 메타크리틱 점수 (0-100점)
- 💬 **User Reviews**: 사용자 리뷰 (최대 10개)

## ⚙️ 주요 설정

```python
# imdb_scraper.py 파일 내부

MAX_CALLS_PER_SECOND = 2    # 초당 요청 수 (기본: 2, 안전)
MAX_RETRIES = 3              # 재시도 횟수
batch_size = 50              # 배치 크기
```

## 📈 성능

| 데이터 수 | 예상 시간 |
|----------|----------|
| 100개    | ~1분     |
| 1,000개  | ~8분     |
| 10,000개 | ~1.5시간 |

*초당 2회 기준 (안전 속도)*

## 📁 출력 파일

- `imdb_data_collected.csv` - 수집된 모든 데이터
- `imdb_data_collected.parquet` - Parquet 형식 (선택)
- `imdb_checkpoint.json` - 진행 상황 저장 (자동)

## 🔍 필터 조건

```python
# vote_count >= 30 AND imdb_id가 존재하는 시리즈만 수집
df_filtered = df[(df['vote_count'] >= 30) & (df['imdb_id'].notna())]
```

## 💡 주요 기능

✅ **비동기 처리**: 빠른 수집 속도  
✅ **Rate Limiting**: 차단 방지  
✅ **자동 재시도**: 실패 시 자동 재시도  
✅ **체크포인트**: 중단 후 이어서 진행  
✅ **실시간 모니터링**: 진행 상황 실시간 확인  

## ⚠️ 주의사항

### 1. IMDB 이용 약관 준수
- 개인 연구/학습 목적으로만 사용
- 상업적 용도 **금지**
- 데이터 재배포 **금지**

### 2. Rate Limiting
- 기본 설정(2회/초)은 안전한 속도
- 속도를 높이면 IP 차단 위험 증가
- 대량 수집 시 며칠에 걸쳐 진행 권장

### 3. HTML 구조 변경
- IMDB가 웹사이트 구조를 변경하면 작동 안 할 수 있음
- 정기적인 코드 업데이트 필요

## 🛠️ 문제 해결

### 429 에러 (Too Many Requests)
```python
# MAX_CALLS_PER_SECOND 값을 낮추세요
MAX_CALLS_PER_SECOND = 1  # 더 느리지만 안전
```

### 중간에 중단됨
```bash
# 다시 실행하면 자동으로 이어서 진행
python imdb_scraper.py
```

### 데이터가 수집되지 않음
1. 브라우저에서 IMDB 페이지 확인
2. HTML 구조 변경 여부 확인
3. 필요시 CSS 선택자 업데이트

## 📝 출력 데이터 예시

```csv
imdb_id,series_title,imdb_rating,imdb_rating_count,meta_score,reviews_json,scraped_at
tt0944947,Game of Thrones,9.2,2150000,91,"[{""title"":""Best show ever""...}]",2024-12-02 10:30:00
tt0903747,Breaking Bad,9.5,1800000,96,"[{""title"":""Masterpiece""...}]",2024-12-02 10:30:15
```

## 🔗 관련 파일

- `imdb_scraper.py` - 메인 크롤러
- `IMDB_SCRAPER_GUIDE.py` - 상세 가이드
- `tv_series_2013_0101_0215_FULL.csv` - 입력 파일 (TMDB 데이터)

## 📚 추가 정보

전체 가이드와 고급 기능은 `IMDB_SCRAPER_GUIDE.py` 파일을 참고하세요.

---

**법적 고지**: 이 도구는 교육 목적으로 제공됩니다. IMDB 이용 약관을 반드시 준수하세요.
