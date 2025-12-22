#“페이지 공통으로 쓰이는 ‘필터/정규화 규칙’만 모아두는 곳”
import pandas as pd

def normalize_content_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    content_type 표준화:
    - 컬럼명 통일: content_type
    - 값 통일: movie / drama (소문자)
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # 컬럼명 후보
    col_candidates = ["content_type", "type", "content"]
    col = next((c for c in col_candidates if c in out.columns), None)

    if col is None:
        return out

    out[col] = out[col].astype(str).str.lower().str.strip()

    type_map = {
        "movie": "movie",
        "film": "movie",

        "tv": "drama",
        "drama": "drama",
        "series": "drama",
        "tv_series": "drama",
    }

    out["content_type"] = out[col].map(type_map)

    # 매핑 실패 제거
    out = out[out["content_type"].notna()]

    return out


# ===========================================================
def normalize_hit_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    hit 라벨 표준화:
    - 컬럼명 통일: hit_label
    - 값 통일: 'Hit' / 'Non-Hit'
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    col_candidates = ["hit_label", "hit_type", "hit"]
    col = next((c for c in col_candidates if c in out.columns), None)

    if col is None:
        return out

    out[col] = out[col].astype(str).str.lower().str.strip()

    hit_map = {
        "hit": "Hit",
        "1": "Hit",
        "true": "Hit",

        "non-hit": "Non-Hit",
        "nonhit": "Non-Hit",
        "0": "Non-Hit",
        "false": "Non-Hit",
    }

    out["hit_label"] = out[col].map(hit_map)
    out = out[out["hit_label"].notna()]  # 매핑 실패 제거

    return out


# ----------------------------------------
# 페이지에서 쓰기 위한 공통 필터 엔진
# ----------------------------------------
def apply_common_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # ✅ 0) content_type 정규화 (가장 먼저!)
    out = normalize_content_type(out)
    out = normalize_hit_label(out)

    # 1) content_type 필터
    ct = filters.get("content_type", "All")
    if ct != "All" and "content_type" in out.columns:
        out = out[out["content_type"] == ct.lower()]

    # 2) year_range
    yr = filters.get("year_range", None)
    if yr and len(yr) == 2:
        start, end = int(yr[0]), int(yr[1])
        for ycol in ["year", "release_year", "review_year"]:
            if ycol in out.columns:
                out[ycol] = pd.to_numeric(out[ycol], errors="coerce")
                out = out[out[ycol].between(start, end, inclusive="both")]
                break

    # 3) hit_type  (normalize_hit_label로 hit_label이 표준화되어 있음)
    ht = filters.get("hit_type", "All")
    if ht != "All" and "hit_label" in out.columns:
        out = out[out["hit_label"] == ht]


    return out.reset_index(drop=True)



