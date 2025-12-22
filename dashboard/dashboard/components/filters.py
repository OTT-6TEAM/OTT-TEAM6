# 분석의 조건
# 같은 데이터로 보고 있다는 걸 강조
# 사이드바 필터를 한 곳에서 관리
# 모든 페이지에 통일한 조건 적용
# 필터 기준 변경 시 한 번만 수정
import streamlit as st

def sidebar_filters():
    """
    대시보드 전 페이지에서 공통으로 사용하는 사이드바 필터
    반환값은 dict 형태
    """

    st.sidebar.header("🔎 Filters")

    content_type = st.sidebar.selectbox(
        "Content Type",
        ["All", "movie", "drama"]
    )


    hit_type = st.sidebar.selectbox(
        "Hit Label",
        ["All", "Hit", "Non-Hit"]
    )

    year_range = st.sidebar.slider(
        "Release Year",
        min_value=2000,
        max_value=2025,
        value=(2000, 2025)
    )
    
    return {
        "content_type": content_type,
        "year_range": year_range,
        "hit_type": hit_type,
    }
    