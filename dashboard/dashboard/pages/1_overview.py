import streamlit as st
from components.filters import sidebar_filters
from components.layout import page_header
from components.style import *
from utils.apply_filters import apply_common_filters

# 1) Sidebar (공통 필터 UI)
filters = sidebar_filters()

# 2) Page Header
page_header("📊 Overview", "Movie & TV Dashboard")

# 3) 데이터 로드 확인(현재 연결돼있는 데이터)
df = "본인 데이터 파일 함수"
df = apply_common_filters(df, filters)

# 4) KPI Row (빈 값 placeholder)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Titles", "-")
with k2:
    st.metric("Hit Rate", "-")
with k3:
    st.metric("Avg Rating", "-")
with k4:
    st.metric("Total Reviews", "-")

st.divider()

# 5) Layout Blocks (차트 자리만 잡기)
left, right = st.columns([2, 1])

with left:
    st.subheader("📈 Yearly Trend")
    st.info("여기에 연도별 트렌드 그래프가 들어갈 예정")

with right:
    st.subheader("🏆 Top 5")
    st.info("여기에 Top 5 작품 리스트/테이블이 들어갈 예정")

st.divider()

st.subheader("🧭 Notes")
st.write(
    "- 지금은 UI 뼈대만 만든 상태\n"
    "- 다음 단계에서 loader로 데이터 로드 → apply_common_filters 적용 → KPI/차트 연결"
)
