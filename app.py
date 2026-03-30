import streamlit as st
from backend import run_pipeline

st.title("Excel to Dashboard AI")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx","xls"])
user_query = st.text_input("Enter your query")

if st.button("Generate"):
    if uploaded_file and user_query:
        result, html = run_pipeline(uploaded_file, user_query)

        st.write("### Result Data")
        if isinstance(result, pd.DataFrame):
            result = result.astype(str)
        st.dataframe(result)

        st.write("### Chart")
        st.components.v1.html(html, height=600)
