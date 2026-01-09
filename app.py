import streamlit as st

st.set_page_config(layout="wide", page_title="Intelligent Data Analyst Assistant")

import pandas as pd
import io
import json
import google.generativeai as genai  # type: ignore
import os
import re
import sys
from io import StringIO
import traceback  # Hata izleme için
import matplotlib  # type: ignore

matplotlib.use('Agg')  # GUI olmayan backend
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import numpy as np  # type: ignore
import sqlite3  # SQL için
from PIL import Image  # type: ignore
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.generativeai")

# KULLANICININ SAĞLADIĞI ANAHTAR (USER-PROVIDED KEY)
PROVIDED_API_KEY =os.getenv("PROVIDED_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Sidebar'da API durumu için yer tutucu
api_status_placeholder = st.sidebar.empty()

try:
    if PROVIDED_API_KEY:
        genai.configure(api_key=PROVIDED_API_KEY)
        GOOGLE_API_KEY = PROVIDED_API_KEY
        api_status_placeholder.success("Google API Key (user-provided) successfully configured.")
    else:
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            api_status_placeholder.warning(
                "WARNING: GOOGLE_API_KEY environment variable not set and not provided in code. AI model will not be accessible.")
        else:
            genai.configure(api_key=GOOGLE_API_KEY)
            api_status_placeholder.success("Google API Key (from environment variable) successfully configured.")
except Exception as e:
    api_status_placeholder.error(f"Error configuring Google API key: {e}.")
    GOOGLE_API_KEY = None

model = None
model_status_placeholder = st.sidebar.empty()
if GOOGLE_API_KEY:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        model_status_placeholder.success("Google Generative Model successfully loaded.")
    except Exception as e:
        model_status_placeholder.error(f"Failed to initialize Google Generative Model: {e}. Check if API key is correct and model is accessible.")
else:
    model_status_placeholder.error("AI model could not be initialized because GOOGLE_API_KEY was not set or provided.")


# --- load_data_from_file function ---
def load_data_from_file(file_obj, separator=';'):
    """
    Loads data from a file object into a Pandas DataFrame and an in-memory SQLite database.
    Supports CSV and Parquet files.
    """
    # Bu fonksiyon Streamlit widget'larından çağrıldığı için st.sidebar.write yerine loglama veya print kullanılabilir
    print("\n--- load_data_from_file function started ---")
    if file_obj is None:
        print(f"Debug: File object is None.")
        return None, None, "Please upload a file.", None, None

    file_name = file_obj.name
    df = None
    conn = None
    error_message = None
    print(f"Debug: File name: {file_name}, Initial separator: '{separator}'")

    try:
        if file_name.lower().endswith('.csv'):
            print(f"Debug: CSV file detected: {file_name}")
            tried_separators = [separator, ',', ';', '\t', '|']
            tried_encodings = ['utf-8', 'iso-8859-9', 'cp1252', 'latin1']
            success = False
            for enc in tried_encodings:
                for sep_try in tried_separators:
                    try:
                        file_obj.seek(0)
                        df = pd.read_csv(file_obj, sep=sep_try, encoding=enc, engine='python', low_memory=False)
                        separator = sep_try
                        print(f"Debug: Success - Separator: '{separator}', Encoding: '{enc}'")
                        success = True
                        break
                    except Exception:
                        pass
                if success:
                    break
            if not success:
                raise Exception(f"Could not read file '{os.path.basename(file_name)}' with supported separators or encodings.")

        elif file_name.lower().endswith('.parquet'):
            print(f"Debug: Parquet file detected: {file_name}")
            file_obj.seek(0)  # Parquet için de seek(0) eklemek iyi bir pratik olabilir
            df = pd.read_parquet(file_obj)
            print(f"Debug: Parquet file successfully read.")
        else:
            error_message = "Unsupported file format. Please upload a .csv or .parquet file."
            print(f"Debug: Unsupported format: {file_name}")
            return None, None, error_message, None, None

        if df is not None and not df.empty:
            print(f"Debug: DataFrame successfully loaded. Rows: {len(df)}, Columns: {len(df.columns)}")
            df_renamed = df.copy()
            try:
                conn = sqlite3.connect(':memory:')
                clean_columns = {col: re.sub(r'\W|^(?=\d)', '_', str(col)) for col in df.columns}
                df_renamed.columns = [clean_columns.get(col, str(col)) for col in df.columns]

                cols = pd.Series(df_renamed.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in
                                                                     range(sum(cols == dup))]
                df_renamed.columns = cols

                df_renamed.to_sql('uploaded_table', conn, if_exists='replace', index=False)
                print(f"Debug: DataFrame loaded into SQLite as 'uploaded_table'. Cleaned columns: {list(df_renamed.columns)}")
            except Exception as e_sql:
                print(f"Debug: Error loading DataFrame to SQLite: {e_sql}")
                conn = None

            schema_info = {
                "file_name": os.path.basename(file_name),
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "original_columns": df.columns.tolist(),
                "sql_columns": list(df_renamed.columns) if conn and hasattr(df_renamed,
                                                                            'columns') else df.columns.tolist(),
                "column_dtypes": {str(col): str(df[col].dtype) for col in df.columns},
                "sql_table_name": "uploaded_table" if conn else None
            }
            schema_str = json.dumps(schema_info, indent=2)
            success_message = f"'{schema_info['file_name']}' successfully loaded ({schema_info['num_rows']} rows, {schema_info['num_columns']} columns)."
            if conn:
                success_message += " Ready for SQL analysis."
            try:
                df_head_markdown = df.head().to_markdown(index=True)
            except Exception:
                df_head_markdown = "```text\n" + df.head().to_string() + "\n```"
            return df, schema_str, success_message, df_head_markdown, conn
        else:
            error_message = "File is empty or could not be read."
            return None, None, error_message, None, None
    except Exception as e:
        error_message = f"Error processing file: {str(e)}"
        traceback.print_exc()
        return None, None, error_message, None, None


# --- Code and SQL Execution Functions ---
def safe_execute_code(code, dataframe, conn_obj):
    print("\n--- safe_execute_code function started ---")
    # print(f"Debug: Python code to execute:\n{code}") # Kod çok uzun olabilir, loglamayı kısaltalım

    local_scope = {
        'df': dataframe, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
        'JSON': json, 'json': json,  # 'json' modülünü küçük harflerle de ekleyin
        'io': io, '_analysis_result_': None, '_plot_object_': None
    }
    global_scope = {
        'JSON': json, 'json': json,

    }
    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirected_output, redirected_error = StringIO(), StringIO()
    sys.stdout, sys.stderr = redirected_output, redirected_error
    executed_successfully, fig_buffer = False, None

    try:
        exec(code, global_scope, local_scope)
        executed_successfully = True
        if local_scope.get('_plot_object_') is not None:
            plot_fig = local_scope['_plot_object_']
            fig_to_save = plot_fig
            if hasattr(plot_fig, 'figure') and isinstance(plot_fig.figure, plt.Figure):
                fig_to_save = plot_fig.figure
            if isinstance(fig_to_save, plt.Figure):
                buf = io.BytesIO()
                fig_to_save.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                fig_buffer = buf
                plt.close(fig_to_save)
                print("Debug: Plot object captured and saved to buffer.")
            else:
                print(f"Debug: _plot_object_ found but not a valid Matplotlib Figure: {type(plot_fig)}")
    except Exception:
        traceback.print_exc(file=redirected_error)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    stdout_output = redirected_output.getvalue().strip()
    stderr_output = redirected_error.getvalue().strip()
    execution_error_msg = stderr_output if not executed_successfully and stderr_output else None
    if executed_successfully and stderr_output:
        stdout_output = f"WARNING:\n```text\n{stderr_output}\n```\n\n" + stdout_output
    analysis_result_obj = local_scope.get('_analysis_result_', None)
    return executed_successfully, stdout_output, execution_error_msg, analysis_result_obj, fig_buffer


def execute_sql_query(sql_code, conn_obj):
    print(f"\n--- execute_sql_query: {sql_code[:100]}... ---")
    if conn_obj is None:
        return False, None, "SQL database connection not found.", None
    try:
        result_df = pd.read_sql_query(sql_code, conn_obj)
        return True, None, None, result_df
    except Exception as e:
        return False, None, str(e), None


# --- Main Interaction Function ---
def process_user_query(message, chat_history, current_df_state, current_schema_state, current_conn_state):
    print(f"\n--- process_user_query (user message): {message[:100]}... ---")
    chat_history.append({"role": "user", "content": message})

    # Streamlit'te anlık yanıt için yer tutucu
    # Bu fonksiyon doğrudan chat_history'i döndürecek, UI'ı st.rerun() ile güncelleyeceğiz.
    # Bu yüzden message_placeholder'a burada gerek yok.

    if current_df_state is None or current_schema_state is None:
        response_content = "Please upload a data file first so I can respond."
        chat_history.append({"role": "assistant", "content": response_content})
        return chat_history
    if model is None:
        response_content = "AI model is not ready. Please ensure your API key is correct and the model is accessible."
        chat_history.append({"role": "assistant", "content": response_content})
        return chat_history

    try:
        schema_dict = json.loads(current_schema_state)
        sql_table_name = schema_dict.get("sql_table_name", "uploaded_table")
        # SQL kolonları df_renamed'den gelmeli, schema_dict'te "sql_columns" olarak saklanıyor
        sql_columns_from_schema = schema_dict.get("sql_columns", [])
        original_columns = schema_dict.get("original_columns", [])
    except Exception as e:
        response_content = f"Error processing data schema: {e}."
        chat_history.append({"role": "assistant", "content": response_content})
        return chat_history

    history_for_prompt = []
    for item in chat_history[:-1]:  # Son kullanıcı mesajı hariç
        if item["role"] == "user":
            history_for_prompt.append(f"User: {item['content']}")
        elif item["role"] == "assistant":
            # Asistan yanıtları bazen çok uzun olabilir, prompt'a eklerken dikkatli olmalı.
            # Şimdilik tamamını ekleyelim.
            history_for_prompt.append(f"Assistant: {item['content']}")

    prompt_parts = [
        "You are an intelligent data analyst assistant who helps the user analyze uploaded datasets and gain insights.",
        "Answer user questions in natural language and in an understandable way.",
        f"The user's data is stored both as a Pandas DataFrame (variable `df`, original column names: {original_columns}) and in an SQLite database in a table named `{sql_table_name}` (column names for SQL: {sql_columns_from_schema}).",
        "If the user requests simple data retrieval, filtering, sorting, or grouping, prefer to generate an **SQLite SQL query** first. Enclose the SQL query within a ```sql\n...\n``` block.",
        "If more complex data manipulation, statistical analysis, or especially a visualization is requested, generate Python (Pandas, Matplotlib, Seaborn) code. Enclose the Python code within a ```python\n...\n``` block.",
        "**Rules for Python Code:**",
        f"  - The Pandas DataFrame is already loaded as `df`. Use original column names: `{original_columns}`",
        "  - Assign the main result (DataFrame, Series, number, etc.) to the variable `_analysis_result_ = ...`.",
        "  - If you are performing visualization (Matplotlib/Seaborn), assign the plot figure to the `_plot_object_` variable, e.g., `_plot_object_ = plt.figure(...)` or `_plot_object_ = sns_plot.figure` (if sns_plot is an AxesSubplot). **DO NOT** use `plt.show()`.",
        "**Rules for SQL Code:**",
        f"  - Table name is `{sql_table_name}`. Column names to use for SQL queries: `{sql_columns_from_schema}`",
        # Düzeltildi
        "Outside the code blocks, provide clear text explaining the code you generated (if necessary) and the results of the analysis to the user.",
        "The general structure of the dataset you are analyzing is as follows:",
        f"File Name: {schema_dict.get('file_name', 'Unknown')}",
        f"Data Schema (JSON for context): {current_schema_state}",  # Schema'nın tamamını gönderebiliriz
        "--- Chat Context (Previous Conversations) ---",
        "\n".join(history_for_prompt),
        "--- User's New Request ---",
        f"User: {message}",
        "--- Your Response (Explanation and, IF NECESSARY, SQL or Python code block) ---",
        "Assistant:",
    ]
    full_prompt = "\n\n".join(prompt_parts)
    # print(f"LLM Prompt (kısaltılmış): {full_prompt[:500]}...")

    llm_full_response_content = ""  # LLM'den gelen tüm metin + execution feedback
    generated_code_type, generated_code = None, None

    message_placeholder = st.chat_message("mr.ai",
                                          avatar="ai").empty()
    current_stream_text = ""

    try:
        response = model.generate_content(full_prompt, stream=True)
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                current_stream_text += chunk.text
                message_placeholder.markdown(current_stream_text + "▌")

        llm_full_response_content = current_stream_text  # Sadece LLM'in ürettiği metin
        message_placeholder.markdown(llm_full_response_content)  # Son hali imleçsiz

        python_code_match = re.search(r"```python\n(.*?)\n```", llm_full_response_content, re.DOTALL)
        sql_code_match = re.search(r"```sql\n(.*?)\n```", llm_full_response_content, re.DOTALL)
        if python_code_match:
            generated_code_type, generated_code = "python", python_code_match.group(1).strip()
        elif sql_code_match:
            generated_code_type, generated_code = "sql", sql_code_match.group(1).strip()
    except Exception as e:
        error_msg = f"Error getting response from AI model: {e}"
        llm_full_response_content += f"\n\n**Hata:** {error_msg}"  # LLM yanıtına hatayı ekle
        message_placeholder.error(llm_full_response_content)  # Hata mesajını göster
        chat_history.append({"role": "assistant", "content": llm_full_response_content})
        return chat_history

    analysis_result_df_output_val, plot_output_image_val, execution_feedback = None, None, ""
    if generated_code:
        if generated_code_type == "python":
            py_success, py_stdout, py_error, py_result_obj, py_fig_buffer = safe_execute_code(generated_code,
                                                                                              current_df_state,
                                                                                              current_conn_state)
            if py_stdout: execution_feedback += f"\n\n---\n**Python Kod Çıktısı (print):**\n```text\n{py_stdout}\n```"
            if py_error: execution_feedback += f"\n\n---\n**Python Kod Çalıştırma Hatası:**\n```text\n{py_error}\n```"
            if py_result_obj is not None:
                analysis_result_df_output_val = py_result_obj if isinstance(py_result_obj,
                                                                            (pd.DataFrame, pd.Series)) else None
                feedback_text_obj = str(py_result_obj) if analysis_result_df_output_val is None else "Tabloda gösteriliyor."
                execution_feedback += f"\n\n---\n**Analiz Sonucu (Python `_analysis_result_`):** {feedback_text_obj}"
            if py_fig_buffer:
                plot_output_image_val = Image.open(py_fig_buffer)
                execution_feedback += "\n\n---\n**Python Görselleştirme Sonucu:** Grafikte gösteriliyor."
        elif generated_code_type == "sql":
            sql_success, _, sql_error, sql_result_df = execute_sql_query(generated_code, current_conn_state)
            if sql_error: execution_feedback += f"\n\n---\n**SQL Çalıştırma Hatası:**\n```text\n{sql_error}\n```"
            if sql_result_df is not None:
                analysis_result_df_output_val = sql_result_df
                execution_feedback += f"\n\n---\n**SQL Sorgu Sonucu:** Tabloda gösteriliyor."
            elif not sql_error:  # Hata yok ama sonuç da yoksa
                execution_feedback += f"\n\n---\n**SQL Sorgu Sonucu:** Sorgu çalıştı ancak bir sonuç döndürmedi."

        llm_full_response_content += execution_feedback
        message_placeholder.markdown(llm_full_response_content)

    chat_history.append({"role": "assistant", "content": llm_full_response_content})
    st.session_state['analysis_result_df'] = analysis_result_df_output_val
    st.session_state['plot_output_image'] = plot_output_image_val
    return chat_history


# --- Streamlit UI Definition ---
st.title("📊 Intelligent Data Analyst Assistant v0.5")
st.markdown(
    "Upload your `.csv` or `.parquet` file. The assistant will generate SQL or Python code to perform analysis and display insights.")

# Initialize session state variables
if 'df_state' not in st.session_state: st.session_state['df_state'] = None
if 'schema_state' not in st.session_state: st.session_state['schema_state'] = None
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'conn_state' not in st.session_state: st.session_state['conn_state'] = None
if 'analysis_result_df' not in st.session_state: st.session_state['analysis_result_df'] = None
if 'plot_output_image' not in st.session_state: st.session_state['plot_output_image'] = None
if 'upload_status_text' not in st.session_state: st.session_state[
    'upload_status_text'] = "File status: No file uploaded yet."
if 'data_preview_markdown' not in st.session_state: st.session_state[
    'data_preview_markdown'] = "First rows will appear here after file upload."
if 'schema_summary_markdown' not in st.session_state: st.session_state[
    'schema_summary_markdown'] = "Schema information will appear here after file upload."
if 'file_uploader_key' not in st.session_state: st.session_state[
    'file_uploader_key'] = 0  # Dosya yükleyiciyi resetlemek için

# Sidebar for file upload and info
with st.sidebar:
    st.header("📂 File Upload & Info")
    # API durumu zaten en üstte gösteriliyor, sidebar'dan kaldırılabilir veya burada da tutulabilir.
    # api_status_placeholder.empty() # Eğer yukarıda tanımlandıysa, sidebar için tekrar tanımlamaya gerek yok
    # model_status_placeholder.empty()

    file_uploader = st.file_uploader("Select Data File (csv,parquet)",
                                     
                                     key=f"file_uploader_{st.session_state['file_uploader_key']}")
    csv_separator_input = st.text_input("CSV Separator (auto-detected)", value=';', placeholder="Usually auto-detected")

    if file_uploader is not None:
        with st.spinner('Loading file...'):
            df, schema_str, success_message, df_head_markdown, conn = load_data_from_file(file_uploader,
                                                                                          csv_separator_input)
            st.session_state['df_state'] = df
            st.session_state['schema_state'] = schema_str
            st.session_state['conn_state'] = conn
            st.session_state['upload_status_text'] = success_message
            st.session_state['data_preview_markdown'] = df_head_markdown
            if schema_str:
                schema_dict = json.loads(schema_str)
                md_text = f"**File:** `{schema_dict.get('file_name', 'N/A')}` ({schema_dict.get('num_rows', 'N/A')}R x {schema_dict.get('num_columns', 'N/A')}C)\n"
                if schema_dict.get("sql_table_name"):
                    md_text += f"**SQL Table:** `{schema_dict.get('sql_table_name')}`\n"
                    md_text += f"   **SQL Columns:** `({', '.join(schema_dict.get('sql_columns', []))})`\n"
                md_text += "**Pandas `df` Columns & Types:**\n"
                for col, dtype in schema_dict.get('column_dtypes', {}).items(): md_text += f"- `{col}` (`{dtype}`)\n"
                st.session_state['schema_summary_markdown'] = md_text
            else:
                st.session_state['schema_summary_markdown'] = success_message  # Hata mesajı varsa gösterir
        st.success(st.session_state['upload_status_text'])

    st.markdown(st.session_state['upload_status_text'])

    with st.expander("🔍 Data Preview (First 5 Rows)", expanded=False):
        if st.session_state['df_state'] is not None:
            st.dataframe(st.session_state['df_state'].head())
        else:
            st.markdown(st.session_state['data_preview_markdown'])

    with st.expander("📊 Data Schema & SQL Table", expanded=False):
        st.markdown(st.session_state['schema_summary_markdown'])

    if st.button("Clear All & Reset File"):
        st.session_state['df_state'] = None
        st.session_state['schema_state'] = None
        st.session_state['chat_history'] = []
        st.session_state['conn_state'] = None
        st.session_state['analysis_result_df'] = None
        st.session_state['plot_output_image'] = None
        st.session_state['upload_status_text'] = "File status: No file uploaded yet."
        st.session_state['data_preview_markdown'] = "First rows will appear here after file upload."
        st.session_state['schema_summary_markdown'] = "Schema information will appear here after file upload."
        st.session_state['file_uploader_key'] += 1  # file_uploader'ı resetle
        st.rerun()

# Main content area
tab1, tab2 = st.tabs(["🤖 Assistant Chat", "📈 Analysis Results & Plots"])

with tab1:
    # Display chat messages from history
    for message_item in st.session_state['chat_history']:  # Değişken adını değiştirdim
        with st.chat_message(message_item["role"],
                             avatar="🧑‍💻" if message_item[
                                                 "role"] == "user" else "🤖"):
            st.markdown(message_item["content"])

    # User input for new messages
    user_input_prompt = st.chat_input("Ask questions about your dataset...")  # Değişken adını değiştirdim
    if user_input_prompt:
        st.session_state['chat_history'] = process_user_query(
            user_input_prompt,
            st.session_state['chat_history'],
            st.session_state['df_state'],
            st.session_state['schema_state'],
            st.session_state['conn_state']
        )
        st.rerun()

with tab2:
    st.header("Analysis Results & Plots")
    if st.session_state['analysis_result_df'] is not None:
        st.subheader("Analysis Result Table")
        if isinstance(st.session_state['analysis_result_df'], pd.DataFrame):
            st.dataframe(st.session_state['analysis_result_df'])
        elif isinstance(st.session_state['analysis_result_df'], pd.Series):
            st.write(st.session_state['analysis_result_df'])
        else:  # Sayı, string vs. ise
            st.markdown(f"**Result:**\n```\n{st.session_state['analysis_result_df']}\n```")
    else:
        st.info("No table results to display yet. Ask the assistant to perform an analysis!")

    if st.session_state['plot_output_image'] is not None:
        st.subheader("Analysis Result Plot")
        st.image(st.session_state['plot_output_image'], caption="Generated Plot", use_column_width=True)
    else:
        st.info("No plot results to display yet. Ask the assistant to generate a visualization!")

st.sidebar.markdown("---")
