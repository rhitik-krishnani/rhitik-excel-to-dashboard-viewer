import streamlit as st
import subprocess
import tempfile
import os

# -------------------------------
# PAGE CONFIG (Light UI)
# -------------------------------
st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

st.title("📊 AI Data Analyst UI")
st.markdown("Upload Excel + Ask Query")

# -------------------------------
# INPUTS
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx", "xls"])
user_query = st.text_input("💬 Enter your query")

run_btn = st.button("🚀 Run Analysis")

# -------------------------------
# RUN BACKEND
# -------------------------------
if run_btn:

    if not uploaded_file or not user_query:
        st.warning("Please upload file and enter query")
        st.stop()

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_file.read())
            temp_file_path = tmp.name

        # Pass inputs as environment variables
        env = os.environ.copy()
        env["USER_QUERY"] = user_query
        env["FILE_PATH"] = temp_file_path

        st.subheader("🧠 Backend Logs")

        # Run backend script
        process = subprocess.Popen(
            ["python", "app_backend.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        output_placeholder = st.empty()

        logs = ""

        # Stream logs live
        for line in process.stdout:
            logs += line
            output_placeholder.code(logs)

        process.wait()

        # Show errors if any
        if process.returncode != 0:
            error = process.stderr.read()
            st.error(error)

        else:
            st.success("✅ Completed")

    except Exception as e:
        st.error(f"Error: {str(e)}")