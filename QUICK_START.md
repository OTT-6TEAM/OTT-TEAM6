# 🎬 IMDB 리뷰 크롤러 - 빠른 시작 가이드

## 📦 제공된 파일

총 **3가지 크롤러** + 가이드 문서

---

## 🚀 크롤러 선택 가이드

### 1️⃣ **평점 + 메타스코어 + 소량 리뷰** 수집
📁 `imdb_scraper.py` (비동기, 빠름)

**수집 데이터:**
- ⭐ IMDB 평점 & 투표수
- 🎯 메타크리틱 점수
- 💬 리뷰 최대 10개 (샘플)

**특징:**
- 가장 빠름 (100개 → 1분)
- 핵심 정보만 수집
- 리소스 적게 사용

**사용:**
```bash
python imdb_scraper.py
```

---

### 2️⃣ **모든 리뷰 수집 - 비동기** (추천 ⭐⭐⭐)
📁 `imdb_full_reviews_async.py` (고성능)

**수집 데이터:**
- 💬 **전체 리뷰** (페이지네이션 자동 처리)
- 📝 제목, 내용, 평점, 작성자, 날짜
- 👍 Helpful 투표수
- ⚠️ Spoiler 여부

**특징:**
- 5-10배 빠름 (병렬 처리)
- 체크포인트 자동 저장
- 대량 수집에 최적

**사용:**
```bash
python imdb_full_reviews_async.py --vote 10
```

---

### 3️⃣ **모든 리뷰 수집 - 동기** (간단함)
📁 `imdb_reviews_sync_improved.py` (개선된 동기)

**수집 데이터:**
- 💬 **전체 리뷰** (2번과 동일)
- 동기 방식으로 안정적

**특징:**
- 코드 이해하기 쉬움
- 소량 데이터에 적합
- 디버깅 편함

**사용:**
```bash
python imdb_reviews_sync_improved.py --vote 10
```

---

## 📊 성능 비교표

| 크롤러 | 속도 | 데이터 | 복잡도 | 추천 상황 |
|-------|-----|--------|--------|----------|
| 1. imdb_scraper | ⚡⚡⚡ | 소량 | ⭐⭐⭐ | 평점만 필요 |
| 2. async (비동기) | ⚡⚡⚡ | 전체 | ⭐⭐ | **대량 리뷰** |
| 3. sync (동기) | ⚡ | 전체 | ⭐⭐⭐⭐ | 소량/학습용 |

---

## 🎯 상황별 추천

### ✅ 평점과 메타스코어만 필요
→ **1번** `imdb_scraper.py` 사용
```bash
python imdb_scraper.py
```

### ✅ 리뷰 전체 필요 + 빠르게 (대량)
→ **2번** `imdb_full_reviews_async.py` 사용
```bash
python imdb_full_reviews_async.py --vote 10
```

### ✅ 리뷰 전체 필요 + 코드 이해 중요
→ **3번** `imdb_reviews_sync_improved.py` 사용
```bash
python imdb_reviews_sync_improved.py --vote 10
```

---

## ⚙️ 패키지 설치

```bash
# 기본 패키지
pip install requests beautifulsoup4 pandas tqdm --break-system-packages

# 비동기 사용 시 추가
pip install aiohttp --break-system-packages
```

---

## 🔧 옵션 설명

### 공통 옵션
```bash
--input, -i    # 입력 CSV 파일 (기본: tv_series_2013_0101_0215_FULL.csv)
--vote, -v     # 최소 vote_count (기본: 10)
--max-pages    # 시리즈당 최대 페이지 수 (None이면 전체)
```

### 예시
```bash
# vote_count >= 30인 시리즈만, 최대 5페이지까지
python imdb_full_reviews_async.py --vote 30 --max-pages 5

# 다른 CSV 파일 사용
python imdb_scraper.py --input my_data.csv
```

---

## 📁 출력 파일

| 크롤러 | CSV 파일 | 내용 |
|-------|---------|------|
| 1번 | `imdb_data_collected.csv` | 평점, 메타스코어, 샘플 리뷰 |
| 2번 | `imdb_reviews_full_async.csv` | 전체 리뷰 (비동기) |
| 3번 | `imdb_reviews_full_sync.csv` | 전체 리뷰 (동기) |

---

## ⏱️ 예상 소요 시간

### 1번 (평점만)
- 100개: ~1분
- 1,000개: ~10분
- 10,000개: ~1.5시간

### 2번 (전체 리뷰, 비동기)
- 100개: ~10분
- 1,000개: ~2시간
- 10,000개: ~20시간

### 3번 (전체 리뷰, 동기)
- 100개: ~50분
- 1,000개: ~8시간
- 10,000개: ~80시간

---

## 🧪 테스트 먼저!

본격 실행 전 소량 테스트 권장:

```bash
# 테스트 스크립트 (3개만)
python test_imdb_scraper.py

# 또는 max-pages로 제한
python imdb_full_reviews_async.py --max-pages 2
```

---

## ⚠️ 주의사항

1. **IMDB 이용약관 준수**
   - 개인 연구/학습 목적만
   - 상업적 사용 금지

2. **Rate Limiting**
   - 초당 2회 이하 권장
   - 과도한 요청 시 IP 차단

3. **체크포인트**
   - 중단되어도 다시 실행 가능
   - 자동으로 이어서 진행

4. **대량 수집**
   - 여러 날에 걸쳐 진행 권장
   - 주기적으로 중간 저장

---

## 📚 추가 문서

- **README_IMDB.md** - 기본 사용법
- **COMPARISON_GUIDE.md** - 상세 비교
- **IMDB_SCRAPER_GUIDE.py** - 완전 가이드

---

## 💡 결론

### 🎯 추천 순서

1. **처음 시작** → `test_imdb_scraper.py`로 테스트
2. **평점만 필요** → `imdb_scraper.py` 실행
3. **리뷰 전체 필요 (대량)** → `imdb_full_reviews_async.py` 실행
4. **리뷰 전체 필요 (소량/학습)** → `imdb_reviews_sync_improved.py` 실행

---

**Happy Crawling! 🚀**

문제가 있으면 에러 메시지를 확인하고, 
각 파일의 상단에 있는 설정값을 조정해보세요.
