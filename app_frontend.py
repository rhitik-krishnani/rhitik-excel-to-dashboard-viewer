import streamlit as st
import streamlit.components.v1 as components
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

        # Streamlit Cloud stores secrets only inside the Streamlit process.
        # Pass the HF key to the backend subprocess via environment variables.
        try:
            hf_api_key = st.secrets["HF_API_KEY"]
        except Exception:
            hf_api_key = os.getenv("HF_API_KEY")

        if not hf_api_key:
            st.error("Missing `HF_API_KEY`. Set it in Streamlit secrets (HF_API_KEY).")
            st.stop()

        env["HF_API_KEY"] = hf_api_key

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
        html_file_path = None

        # Stream logs live
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.startswith("HTML_FILE_PATH="):
                html_file_path = line.split("HTML_FILE_PATH=", 1)[1].strip()
                continue

            logs += line + "\n"
            output_placeholder.code(logs[-40000:])

        process.wait()

        # Show errors if any
        if process.returncode != 0:
            error = process.stderr.read()
            st.error(error)

        else:
            if html_file_path and os.path.exists(html_file_path):
                try:
                    with open(html_file_path, "r", encoding="utf-8") as f:
                        html_code = f.read()

                    st.subheader("📈 Generated Chart")
                    components.html(html_code, height=650, scrolling=True)
                finally:
                    try:
                        os.remove(html_file_path)
                    except Exception:
                        pass

            st.success("✅ Completed")

    except Exception as e:
        st.error(f"Error: {str(e)}")
