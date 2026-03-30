import json
import os
import re
import tempfile
from textwrap import dedent
import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import HTTPException

print("Imported all header files sucessfully !!", flush=True)

def get_table_metadata(df, table_name, n=5):
    df_copy = df.head(n).copy()
    
    # Convert datetime columns to string
    for col in df_copy.select_dtypes(include=['datetime', 'datetime64[ns]']).columns:
        df_copy[col] = df_copy[col].astype(str)
    
    return {
        "table_name": table_name,
        "columns": df.columns.tolist(),
        "sample_records": df_copy.to_dict(orient='records')
    }

load_dotenv()
HF_API_KEY = st.secrets.get("HF_API_KEY") or os.getenv("HF_API_KEY")
HF_API_KEY = st.secrets.get("HF_API_KEY") or os.getenv("HF_API_KEY")

MODEL = "Qwen/Qwen2.5-72B-Instruct"
API_URL = f"https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}


def narrate(system_prompt, user_prompt):
    try:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": dedent(system_prompt).strip()},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1600
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            raise Exception(response.text)

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")

def get_tables_selection_prompt(user_query, metadata_json):
    system_template = """
You are an expert data analyst responsible for selecting the most relevant tables for a given user query.
You carefully analyze table metadata including column names and sample records before making a decision.
"""

    prompt_template = f"""
User Query:
{user_query}

Available Tables Metadata (JSON):
{metadata_json}

Instructions:
1. Each top-level key in the metadata JSON represents a table name.
2. Carefully analyze the user query.
3. Select ONLY those tables that are relevant to answering the query.
4. Use column names and sample records to make an informed decision.
5. Prefer tables that contain relevant keywords (e.g., 'region', 'sales', 'date', etc.).
6. Do NOT guess — select only if there is clear relevance.

Output Rules (STRICT):
- Return ONLY a valid JSON list of table names (top-level keys).
- Do NOT include explanations, text, or formatting.
- Do NOT include anything except the JSON list.

Output Format Examples:
["orders_df", "returns_df"]

If no tables are relevant:
[]

Important:
- Table names MUST exactly match the top-level keys from the metadata JSON.
- Output must be strictly valid JSON.
"""

    return system_template, prompt_template

def get_pandas_code_prompt(user_query, selected_metadata_json, question_response):

    system_template = """
You are an expert in generating executable pandas code to answer user queries.
You have access only to the dataframes listed in the metadata provided.
Your goal is to generate clean, correct, and executable Python pandas code.
"""

    prompt_template = f"""
You will be provided with:
1. User Query
2. Selected Tables Metadata
3. Available DataFrames

## User Query:
{user_query}

## Selected Tables Metadata:
{selected_metadata_json}

## Available DataFrames (STRICT - use only these exact names):
{list(json.loads(selected_metadata_json).keys())}

### Strict Instructions for Code Generation:

1. Use ONLY these dataframes: {question_response}
2. The code MUST be executable using exec()
3. ALWAYS store final output in a variable named: result
4. Do NOT use print statements
5. Do NOT include explanations or comments
6. Do NOT create new dataframes unless required
7. Ensure column names match EXACTLY from metadata
8. Prefer pandas operations like groupby, sum, agg, etc.

### Output Format (STRICT):
<python_code>
result = ...
</python_code>

### Example:
<python_code>
result = Orders_df.groupby("Region")["Sales"].sum().reset_index()
</python_code>

Important:
- Output ONLY valid python code inside <python_code> tags
- No extra text before or after
"""

    return system_template, prompt_template

def get_html_chart_code_prompt(user_query, result_df_json):

    system_template = """
You are an expert data visualization engineer.
Generate complete, valid, self-contained HTML that renders an interactive dashboard using Plotly.js.

CRITICAL OUTPUT RULES (STRICT):
- Output ONLY raw HTML (one complete document).
- No explanations, no markdown, no code fences, no surrounding text.
- No external CSS frameworks (no Tailwind/Bootstrap). Inline CSS in a <style> tag only.
- Use Plotly.js via CDN <script> tag.
"""

    prompt_template = f"""
You are provided:

USER QUERY:
{user_query}

DATA (JSON):
{result_df_json}

STRICT INSTRUCTIONS:

1) Output must be a full HTML document:
   - Include: <meta charset="utf-8">, <meta name="viewport" content="width=device-width, initial-scale=1">
   - Use a modern, clean UI: neutral background, card layout, consistent spacing, readable fonts.

2) Visual selection (automatic, must be sensible):
   - Analyze the USER QUERY + DATA shape and pick the best visualization(s).
   - Prefer:
     - Time series -> line/area (with proper x-axis type)
     - Category vs value -> bar (sorted if appropriate)
     - Part-to-whole -> donut/pie ONLY if <= 8 categories else bar
     - Distribution -> histogram/box
     - Correlation -> scatter with trendline-like styling if helpful (no stats libs required)
     - Multiple metrics -> grouped bar / multi-line with legend
     - Many categories -> horizontal bar, limit + “Top N” if needed
   - If the DATA is already an aggregated table suitable for display, use a table and optionally a chart.

3) Edge cases (must handle, no crashing):
   - If DATA is empty / not parseable / not a list of records / or has no usable numeric columns:
     - Render a nice “No chart available” card and show a fallback HTML table (if any rows) or a short message.
   - Handle nulls: omit null points or treat as gaps; do not render NaN labels.
   - If there are too many rows for a chart (> 5000), downsample or aggregate sensibly; also show a note inside the UI (not as plain text outside HTML).

4) Plotly requirements (must follow):
   - Use Plotly.js from CDN (e.g., https://cdn.plot.ly/plotly-2.27.0.min.js or latest stable).
   - Use responsive rendering: Plotly.newPlot(..., {{responsive: true}}) and CSS that allows resizing.
   - Provide: clear title, axis titles, legend (when multiple series), hover tooltips with formatted values.
   - Number formatting:
     - Use thousands separators.
     - Use compact formatting for big numbers where appropriate (K/M/B).
     - Use 0–2 decimals by default; more only if needed.
     - If query implies currency/percent, format accordingly (e.g. $ with separators, % with 1–2 decimals).
   - Consistent styling:
     - Use a modern font stack (system UI).
     - Use a cohesive color palette; avoid neon; ensure contrast.
     - Use `template: "plotly_white"` or equivalent clean theme.
     - Gridlines subtle; margins balanced.

5) Dashboard UI layout (must look consistent and modern):
   - Full-viewport responsive layout (fits desktop and mobile).
   - Use cards with padding, rounded corners, light shadow/border.
   - Include a header area with the query as a subtitle.
   - The chart lives in a card with fixed min-height and scales responsively.
   - Optional: a “Data preview” card showing a compact table (first 20 rows) with sticky header and scroll.

6) Implementation constraints (strict):
   - No external assets besides Plotly CDN.
   - No inline base64 images.
   - No eval of arbitrary code. Only parse the provided DATA JSON safely in JS.
   - Do not assume specific column names; infer roles:
     - Candidate x: datetime-like strings, or first non-numeric column
     - Candidate y: numeric columns
     - Candidate series/group: low-cardinality categorical column

OUTPUT FORMAT (STRICT):
Return ONLY:
<html>
  <head>...<style>...</style>...<script src="...plotly..."></script></head>
  <body>...<div id="chart"></div>...<script>/* parse data; choose chart; render */</script></body>
</html>

ABSOLUTE RULE:
- Do NOT include any text before <html> or after </html>.
"""

    return system_template, prompt_template

## complete pipeline

file_path = os.getenv("FILE_PATH")
print(f"Step 1 ---> Read file path that is , {file_path}", flush=True)
user_query = os.getenv("USER_QUERY")
print(f"Step 2 ---> Read User query that is , {user_query}", flush=True)

sheets_dict = pd.read_excel(file_path, sheet_name=None)

dataframes = {
    f"df{i+1}": df 
    for i, (sheet_name, df) in enumerate(sheets_dict.items())
}

globals().update(dataframes)

print("Step 3 ---> Dataframe created from the excel file read !!", flush=True)
print(dataframes.keys(), flush=True)

metadata_dict = {
    table_name: get_table_metadata(df, table_name)
    for table_name, df in dataframes.items()
}
metadata_json = json.dumps(metadata_dict, indent=2)
print("Step 4 ---> metadata_json created from the dataframes !!", flush=True)
print(", ".join(json.loads(metadata_json).keys()), flush=True)

system_prompt, prompt = get_tables_selection_prompt(user_query, metadata_json)
question_response = narrate(system_prompt, prompt)

print("Step 5 ---> dataframe selection as per user query is below !!", flush=True)
print(question_response, flush=True)

if isinstance(question_response, str):
    question_response = json.loads(question_response)

metadata_lower_map = {k.lower(): k for k in metadata_dict.keys()}

selected_metadata = {
    metadata_lower_map[t.lower()]: metadata_dict[metadata_lower_map[t.lower()]]
    for t in question_response
    if t.lower() in metadata_lower_map
}

selected_metadata_json = json.dumps(selected_metadata, indent=2)

print("Step 6 ---> selected_metadata_json as per user query is below !!", flush=True)
print(", ".join(json.loads(selected_metadata_json).keys()), flush=True)


system_prompt, prompt = get_pandas_code_prompt(
    user_query,
    selected_metadata_json,
    question_response
)

query_pandas_code = narrate(system_prompt, prompt)

code_to_execute = re.search(
    r"<python_code>\s*(.*)\s*</python_code>",
    query_pandas_code,
    re.DOTALL
).group(1)

# exec(code_to_execute)
for name, df in dataframes.items():
    globals()[name] = df

exec(code_to_execute, globals())

print("Step 7 ---> result of pandas code given by LLM is below !!", flush=True)
print(result, flush=True)


html_code_query_code = None
system_prompt, prompt = get_html_chart_code_prompt(user_query, result)
print("Step 8 ---> HTML code given by LLM is below !!", flush=True)
html_code_query_code = narrate(system_prompt, prompt)

print("Step 9 ---> All steps completed. Preparing HTML output for Streamlit UI.", flush=True)

with tempfile.NamedTemporaryFile(
    delete=False, suffix=".html", mode="w", encoding="utf-8"
) as f:
    f.write(html_code_query_code)
    html_file_path = f.name

print(f"HTML_FILE_PATH={html_file_path}", flush=True)
