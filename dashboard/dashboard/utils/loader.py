# utils/loader.py
# 1. 캐시를 데이터 단위로 관리하려고 -> 파일은 한 번만 읽고 재사용
# 2. 데이터 계약서 역할
# 3. 페이지 코드 쉽게 읽을려고
# 4. 협업에서 사고 안 나게 방지
import pandas as pd
import streamlit as st


# ========== Page 1 (Overview) ==========

# ========== Page 2 승윤 (Synopsis TF-IDF) ==========

# ========== Page 3 예원 (BERTopic) ==========

# ========== Page 4 승윤(Review Analysis) ==========
@st.cache_data 
def load_review_tfidf_keywords():
    return pd.read_parquet("data/03_review/review_tfidf_keywords_all.parquet") #Review TF-IDF Keywords

@st.cache_data
def load_review_topic_summary():
    return pd.read_parquet("data/03_review/review_topic_summary_30w_all.parquet") #Review Topic Summary (전체)

@st.cache_data
def load_review_topic_summary_by_type():
    return pd.read_parquet("data/03_review/review_topic_summary_30w_by_type.parquet") #Review Topic Summary (타입별: Movie / TV)
# ========== Page 5 지연 (Prediction) ==========
