import streamlit as st
import pandas as pd
from app_backend import run_pipeline   # we will create this function

st.set_page_config(layout="wide")

st.title("📊 Excel AI Analyzer")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
user_query = st.text_input("Enter your query")

if st.button("Run Analysis"):

    if uploaded_file is None or user_query.strip() == "":
        st.warning("Please upload file and enter query")
        st.stop()

    # Run backend pipeline
    result_df, html_chart = run_pipeline(uploaded_file, user_query)

    # Show results
    st.subheader("📄 Result Data")
    st.dataframe(result_df)

    st.subheader("📊 Visualization")
    st.components.v1.html(html_chart, height=800, scrolling=True)
