import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page


page = st.sidebar.selectbox("Do you want to explore Or make a prediction?", ('I want to explore', 'I want to predict'))

if page == "I want to explore":
    show_explore_page()
else:
    show_predict_page()
