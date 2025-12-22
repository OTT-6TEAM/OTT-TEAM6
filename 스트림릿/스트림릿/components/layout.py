# 화면의 뼈대 -> 모든 페이지를 같은 구성으로 제목 만들기
import streamlit as st

def page_header(title : str, subtitle : str | None =  None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)