import streamlit as st

from components.filters import sidebar_filters
from components.layout import page_header
from components.style import *
from utils.apply_filters import apply_common_filters
from utils.loader import (
    load_review_tfidf_keywords,
    load_review_topic_summary,
    load_review_topic_summary_by_type,
)

# 1) Sidebar (공통 필터)
filters = sidebar_filters()

# 2) Header
page_header("💬 Review", "리뷰 기반 서사/토픽 탐색")

# 3) 데이터 로드 확인(현재 연결돼있는 데이터)
df = load_review_topic_summary_by_type()
df = apply_common_filters(df, filters)

st.subheader("🔎 데이터 로드 확인")
st.write(df.head())
st.write("row:", len(df))

st.divider()

# 4) KPI(일단 자리)
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Total Reviews", "-")
with k2:
    st.metric("Avg Sentiment", "-")
with k3:
    st.metric("Hit vs Non-Hit Gap", "-")

st.divider()


# 5) 메인 레이아웃
left, right = st.columns([2, 1])

with left:
    st.subheader("📌 Topic Summary (전체/타입별)")
    st.info("여기에 토픽별 리뷰 수/감성/흥행 라벨 분포 요약 테이블이 들어갈 예정")
    st.write("placeholder")

    st.divider()

    st.subheader("🧾 Narrative Keywords")
    st.info("여기에 토픽별 서사 키워드(Top N) 또는 워드클라우드/바차트가 들어갈 예정")
    st.write("placeholder")

with right:
    st.subheader("🎛️ Topic Selector (페이지 전용)")
    st.info("여기에 토픽 선택 UI (selectbox) 들어갈 예정")
    st.write("placeholder")

    st.divider()

    st.subheader("📝 Topic Interpretation")
    st.info("선택된 토픽의 요약/라벨링/예시 문장(샘플) 영역")
    st.write("placeholder")

st.divider()

st.subheader("📎 Notes")
st.write(
    "- 리뷰 토픽은 ‘성공 원인 단서’ 탐색용\n"
    "- 과해석 방지: 토픽은 묶음의 경향성, 정답이 아님"
)
