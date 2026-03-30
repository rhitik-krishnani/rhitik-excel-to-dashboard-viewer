import pandas as pd
import json
from requests_aws4auth import AWS4Auth
import re
import httpx
import random
import os
from textwrap import dedent
from fastapi import HTTPException
from fastapi import HTTPException
import streamlit as st
from dotenv import load_dotenv
import requests
print("Imported all header files sucessfully !!")

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
            "max_tokens": 2000
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)

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
Generate complete, valid, self-contained HTML with inline Plotly.js that best visualizes the given data.
Output ONLY raw HTML. No explanations. No markdown. No code blocks.
"""

    prompt_template = f"""
You are provided with below :

### USER QUERY ###
{user_query}

### DATA (JSON) ###
{result_df_json}

### REQUIREMENTS (STRICT — MUST FOLLOW ALL)

1. Output ONLY raw HTML.
   - Do NOT include explanations, markdown, or code blocks.
   - Response must start with <html> and end with </html>.

2. The HTML must be COMPLETE and EXECUTABLE:
   - Include <html>, <head>, and <body> tags.
   - Include Plotly.js via CDN using a <script> tag.
   - Include exactly one chart container <div> with a unique id.
   - Include a <script> block that defines data, layout, and calls Plotly.newPlot().

3. Chart Selection:
   - Automatically select the MOST APPROPRIATE chart type based on the data and user query.
   - Do NOT ask for clarification.
   - Use only one chart.

4. Data Handling:
   - Use the provided JSON data exactly as given.
   - Handle null, undefined, or missing values safely (filter or replace appropriately).
   - Ensure no runtime JavaScript errors.

5. Visualization Best Practices:
   - Include meaningful chart title, axis labels, and legend (if applicable).
   - Use clean and readable formatting (fonts, spacing, margins).
   - Apply proper number formatting (round values where appropriate).
   - Use visually appealing and consistent colors.

6. Responsiveness (MANDATORY):
   - Chart must fill the entire screen.
   - Use responsive layout (e.g., 100vw, 100vh, autosize: true).
   - Ensure chart resizes properly with window size.

7. Code Quality:
   - Ensure all JavaScript objects and brackets are properly closed.
   - No unused variables.
   - No syntax errors.
   - Plot must render successfully without modification.

### OUTPUT FORMAT — STRICTLY FOLLOW ###
<html>
...
</html>

IMPORTANT : ENSURE NOT TO INCLUDE any extra text or commentary. Follow the OUTPUT FORMAT.
"""

    return system_template, prompt_template


def run_pipeline(uploaded_file, user_query):

    file_path = uploaded_file
    print(f"Step 1 ---> Read file path that is , {file_path}")
    print(f"Step 2 ---> Read User query that is , {user_query}")

    sheets_dict = pd.read_excel(file_path, sheet_name=None)

    dataframes = {
        f"df{i+1}": df 
        for i, (sheet_name, df) in enumerate(sheets_dict.items())
    }

    globals().update(dataframes)

    print("Step 3 ---> Dataframe created from the excel file read !!")
    print(dataframes.keys())

    metadata_dict = {
        table_name: get_table_metadata(df, table_name)
        for table_name, df in dataframes.items()
    }
    metadata_json = json.dumps(metadata_dict, indent=2)
    print("Step 4 ---> metadata_json created from the dataframes !!")
    print(", ".join(json.loads(metadata_json).keys()))

    system_prompt, prompt = get_tables_selection_prompt(user_query, metadata_json)
    question_response = narrate(system_prompt, prompt)

    print("Step 5 ---> dataframe selection as per user query is below !!")
    print(question_response)

    if isinstance(question_response, str):
        question_response = json.loads(question_response)

    metadata_lower_map = {k.lower(): k for k in metadata_dict.keys()}

    selected_metadata = {
        metadata_lower_map[t.lower()]: metadata_dict[metadata_lower_map[t.lower()]]
        for t in question_response
        if t.lower() in metadata_lower_map
    }

    selected_metadata_json = json.dumps(selected_metadata, indent=2)

    print("Step 6 ---> selected_metadata_json as per user query is below !!")
    print(", ".join(json.loads(selected_metadata_json).keys()))


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

    for name, df in dataframes.items():
        globals()[name] = df

    exec(code_to_execute, globals())

    print("Step 7 ---> result of pandas code given by LLM is below !!")
    print(result)

    html_code_query_code = None
    prompt,system_prompt = get_html_chart_code_prompt(user_query, result)
    print("Step 8 ---> HTML code given by LLM is below !!")
    html_code_query_code = narrate(system_prompt, system_prompt)

    print("Step 9 ---> All 8 steps completed !!")

    return result, html_code_query_code
