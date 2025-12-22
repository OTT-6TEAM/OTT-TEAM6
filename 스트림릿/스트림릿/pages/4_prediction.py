import streamlit as st
from components.filters import sidebar_filters
from components.layout import page_header
from components.style import *
from utils.apply_filters import apply_common_filters

# 1) Sidebar (공통 필터)
filters = sidebar_filters()

# 2) Header
page_header("🔮 Prediction", "흥행 예측 · 시뮬레이션")

# 3) 데이터 로드 확인(현재 연결돼있는 데이터)
df = "본인 데이터 파일 함수"
df = apply_common_filters(df, filters)

# 4) 상단 요약 영역 (자리만)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Predicted Hit Probability", "-")
with c2:
    st.metric("Prediction Label", "-")
with c3:
    st.metric("Model Confidence", "-")

st.divider()

# 5) 메인 레이아웃
left, right = st.columns([2, 1])

with left:
    st.subheader("🧪 Simulation Input")
    st.info(
        "여기에 줄거리 입력 / 주요 속성 선택 등\n"
        "사용자 시뮬레이션 입력 UI가 들어갈 예정"
    )
    st.write("placeholder")

    st.divider()

    st.subheader("📈 Prediction Result")
    st.info(
        "여기에 예측 결과 시각화\n"
        "(확률 바, 게이지, 설명 텍스트 등)가 들어갈 예정"
    )
    st.write("placeholder")

with right:
    st.subheader("⚙️ Model Info")
    st.info(
        "사용한 모델 정보\n"
        "- 모델 종류\n"
        "- 학습 데이터 범위\n"
        "- 주요 특징"
    )
    st.write("placeholder")

    st.divider()

    st.subheader("📝 Interpretation Guide")
    st.info(
        "예측 결과 해석 가이드\n"
        "- 확률은 절대값이 아님\n"
        "- 비교/참고용 지표"
    )
    st.write("placeholder")

st.divider()

st.subheader("📎 Notes")
st.write(
    "- 예측 모델은 의사결정 보조 도구\n"
    "- 흥행은 외부 요인(마케팅, 시기 등)에 크게 영향받음"
)
