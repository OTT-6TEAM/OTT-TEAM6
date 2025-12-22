import streamlit as st
from components.filters import sidebar_filters
from components.layout import page_header
from components.style import *
from utils.apply_filters import apply_common_filters

# 1) Sidebar (공통 필터)
filters = sidebar_filters()

# 2) Header
page_header("📝 Synopsis", "TF-IDF 키워드 + BERTopic 토픽 탐색")

# 3) 데이터 로드 확인(현재 연결돼있는 데이터)
df = "본인 데이터 파일 함수"
df = apply_common_filters(df, filters)

# 4) 섹션 구성 (자리만)
tab1, tab2 = st.tabs(["🧾 TF-IDF", "🧠 BERTopic"])

with tab1:
    st.subheader("🧾 TF-IDF Keywords")
    st.info("여기에 흥행/비흥행(또는 전체) TF-IDF 키워드 표/차트가 들어갈 예정")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### ✅ Hit Keywords (Top N)")
        st.write("placeholder")
    with c2:
        st.markdown("### ❌ Non-Hit Keywords (Top N)")
        st.write("placeholder")

    st.divider()

    st.subheader("📌 해석 메모")
    st.write("- 키워드는 ‘설명 변수’가 아니라 ‘단서’로 해석\n- 장르/시대별 편향 가능성 존재")

with tab2:
    st.subheader("🧠 BERTopic Topics")
    st.info("여기에 토픽 요약 테이블/분포/대표 키워드가 들어갈 예정")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### 📊 Topic Overview")
        st.write("placeholder (토픽별 비중/점수/분포)")

    with right:
        st.markdown("### 🔍 Topic Detail")
        st.write("placeholder (토픽 선택 → 대표 키워드/예시)")

    st.divider()

    st.subheader("📌 해석 메모")
    st.write("- 토픽 이름은 사람이 붙이는 ‘라벨’\n- 토픽은 완벽하지 않고, 묶음의 경향성을 보여줌")
