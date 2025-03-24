
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import ast
import numpy as np
from datetime import datetime
import re



# Authentication functions
def authenticate_user(username, password):
    try:
        credentials = st.secrets["credentials"]
        if username in credentials and credentials[username] == password:
            return True
        return False
    except Exception as e:
        st.error(f"Error reading credentials: {e}")
        return False

def login_page():
    st.title(":brain: LLM-Powered Diseases Analysis")
    st.title(":desktop_computer: Login Page")
    username = st.text_input(":alien: Username")
    password = st.text_input(":sleuth_or_spy: Password", type='password')
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state['authenticated'] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Validation functions
def validate_python_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        st.error(f"Syntax error in generated code: {e}")
        return False

def is_count_query(query):
    keywords = ["count", "number of", "how many", "total"]
    return any(keyword in query.lower() for keyword in keywords)

def extract_code_from_response(raw_response):
    code = raw_response.strip()
    
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].strip()
    elif "import" in code:
        code = code[code.find("import"):]
    
    code = code.replace("fig.show()", "st.plotly_chart(fig)")
    
    return code

# Data loading and preprocessing
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('Patient Data Updated.csv')
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        now = datetime.now()
        df['age'] = df['birthdate'].apply(
            lambda x: now.year - x.year - ((now.month, now.day) < (x.month, x.day)) 
            if pd.notna(x) else np.nan
        )

        totalPatients = df.copy()
        
        df = df[df['ICD_code'] != '0']
        
        df.loc[:, 'ICD_code'] = df['ICD_code'].astype(str).apply(lambda x: x.split(','))
        df = df.explode('ICD_code')
        df['ICD_code'] = df['ICD_code'].fillna('Unknown')
        
        df[['Codes', 'Diseases']] = df['ICD_code'].str.extract(r'\(([^)]+)\)\s*(.*)')
        df['Diseases'] = np.where(df['Codes'].isna(), df['ICD_code'], df['Diseases'])
        
        df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
        
        df_unspecified = df[df['Codes'].isna()]
        df_clean = df[~df['Codes'].isna()]
        
        return df, df_unspecified, df_clean, totalPatients
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def get_age_group(age):
    if age < 20: return '0-20'
    elif age < 30: return '20-29'
    elif age < 50: return '30-49'
    else: return '50+'

def run_llm_visualization(query, data,option):
    load_dotenv()
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    if not groq_api_key:
        st.error("GROQ API key not found. Please check your .env file.")
        return None
    
    # Initialize LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=option, temperature=0.2)
    
    # Create prompt
    prompt = f"""
    Generate Python code for a Plotly visualization in Streamlit based on: {query}
    
    Dataframe variable name: 'data' (already loaded)
    
    Requirements:
    1. Use either plotly.graph_objects (as go) or plotly.express (as px)
    2. Create figure as 'fig = go.Figure(...)' or 'fig = px.bar(...)' etc.
    3. Display using EXACTLY: st.plotly_chart(fig)
    4. Do NOT include any other code, imports, or explanations
    5. Never use plt.show() or fig.show()
    6. Use only these columns: {', '.join(data.columns)}
    7. Make sure to handle any empty data gracefully with appropriate error checks
    8. Use color for better visualization
    9. Add appropriate titles and labels
    
    Data sample (first 5 rows):
    {data.head(5).to_string()}
    """
    
    # Get response from LLM
    try:
        response = llm.invoke(prompt)
        raw_code = response.content if hasattr(response, 'content') else str(response)
        return extract_code_from_response(raw_code)
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return None



def execute_visualization_code(code, data):
    """Execute the visualization code with proper error handling"""
    if not code:
        return False
        
    try:
        # Validate syntax
        if not validate_python_syntax(code):
            raise SyntaxError("Invalid Python syntax in generated code")
        

        required = ["fig", "st.plotly_chart"]
        if not all(req in code for req in required):
            missing = [req for req in required if req not in code]
            raise ValueError(f"Missing required elements: {', '.join(missing)}")
        

        exec_globals = {
            'pd': pd,
            'go': go,
            'px': px,
            'st': st,
            'data': data,
            'np': np,
            '__builtins__': __builtins__
        }
        
        exec(code, exec_globals)
        st.success("Visualization rendered successfully!")
        return True
        
    except Exception as e:
        st.error(f"Execution error: {str(e)}")
        st.subheader("Common fixes:")
        st.write("- Check column names match your data")
        st.write("- Ensure query specifies existing numerical/categorical columns")
        st.write("- Try being more specific in your query")
        st.write(f"- Available columns: {', '.join(data.columns)}")
        return False

def agent_tab(df_clean):
    #st.title("Medical Data Visualization")
    

    if "query_result" not in st.session_state:
        st.session_state.query_result = None
    

    try:
        data = df_clean[['id', 'ICD_code', 'age']].copy()
        data.rename(columns={'id': 'patient_id', 'ICD_code': 'Diseases'}, inplace=True)
        

        st.markdown("### Available data columns:")
        st.info(f"You can query about: {', '.join(data.columns)}")
        
        query = st.text_input("Enter your query (e.g., 'top 10 diseases bar chart and pie chart','top 10 diseases pie chart', 'show distribution of age groups')")
        
        col1, col2, col3 = st.columns([1, 1,4])
        with col1:
            submit_button = st.button(":sparkles: Generate Visualization", use_container_width=True)
        with col2:
            clear_button = st.button(":wastebasket: Clear Results", use_container_width=True)
        with col3:
            option = st.selectbox(
                "Select LLM Model to try",
                ("llama3-70b-8192", "llama-3.3-70b-versatile", "llama-3.1-8b-instant","llama-guard-3-8b","llama3-8b-8192","distil-whisper-large-v3-en","gemma2-9b-it","mixtral-8x7b-32768","whisper-large-v3","whisper-large-v3-turbo"),
                index=0,
                placeholder="Select contact method...",
            )
            
        if clear_button:
            st.session_state.query_result = None
            st.rerun()
            
        if submit_button and query:
            if is_count_query(query):

                st.session_state.query_result = f"Count Result: {data.shape[0]}"
                st.write(st.session_state.query_result)
            else:
                with st.spinner("Generating visualization..."):

                    code = run_llm_visualization(query, data,option)
                    
                    if code:
                        # Display the generated code
                        #with st.expander("View Generated Code", expanded=False):
                            #st.code(code, language='python')
                        
                        success = execute_visualization_code(code, data)
                        
                        if success:
                            st.session_state.query_result = code
        
            
    except Exception as e:
        st.error(f"Error in visualization tab: {str(e)}")

def main_app():
    st.set_page_config(page_title="LLM-Powered Diseases Analysis Dashboard", layout="wide")
    st.title(":brain: LLM-Powered Diseases Analysis")


    st.header(":bar_chart: Age Filters")
    min_age, max_age = st.slider("Select Age Range", 0, 105, (10, 50), step=10)


    df, df_unspecified, df_clean, totalPatients = load_and_process_data()

    if df is None or df_clean is None:
        st.error("Failed to load data. Please check your data file.")
        st.stop()

    totalPatients = totalPatients[(totalPatients['age'] >= min_age) & (totalPatients['age'] <= max_age)]
    df = df[(df['age'] >= min_age) & (df['age'] <= max_age)]
    df_unspecified = df_unspecified[(df_unspecified['age'] >= min_age) & (df_unspecified['age'] <= max_age)]
    df_clean = df_clean[(df_clean['age'] >= min_age) & (df_clean['age'] <= max_age)]

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Overview"


    def switch_tab(tab_name):
        st.session_state.active_tab = tab_name



    tabs = ["Overview", "Age Analysis", "Disease Distribution", "Comorbidity Matrix", "Ask Agent", "Second Dashboard"]
    selected_tab = st.radio("Navigate to:", tabs, index=tabs.index(st.session_state.active_tab), key="tab_selector",horizontal=True)

    switch_tab(selected_tab)

    # Tab 1: Overview
    
    if st.session_state.active_tab == "Overview":
        with st.spinner("Loading Overview..."):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total number of Patients", len(totalPatients))
            col2.metric("Patients with unspecified ICD10 Code", len(totalPatients) - len(df_clean['id'].unique()))
            col3.metric("Patient with ICD10 Code", len(df_clean['id'].unique()))
            col4.metric("Patients with 2 or more ICD10 codes", len(df_clean['id'].value_counts()[lambda x: x >= 2]))

            col_raw, col_chart = st.columns(2)
            with col_raw:
                st.subheader("Raw Data")
                st.dataframe(df_clean[["id", "ICD_code", "age"]])
            
            with col_chart:
                st.subheader("Patient Distribution by Age Group")
                age_groups = df['age'].apply(get_age_group).value_counts().reset_index()
                age_groups.columns = ['Age Group', 'Number of Patients']
                fig = px.bar(age_groups, x='Age Group', y='Number of Patients', 
                            title="Patient Distribution by Age Group", color='Age Group')
                st.plotly_chart(fig)
            
            st.subheader("Top 20 Diseases")
            top_diseases = df_clean['ICD_code'].value_counts().nlargest(20).reset_index()
            top_diseases.columns = ['ICD Code', 'Number of Patients']
            fig = px.bar(top_diseases, x='ICD Code', y='Number of Patients', 
                        title="Top 20 Diseases by Number of Patients", color='ICD Code', height=800)
            fig.update_layout(xaxis=dict(tickangle=-45))
            st.plotly_chart(fig)

    # Tab 2: Age Analysis
    elif st.session_state.active_tab == "Age Analysis":
        with st.spinner("Loading Age Analysis..."):
            col_Age1, col_Age2 = st.columns(2)
            with col_Age1:
                age_groups = df_clean['age'].apply(get_age_group).value_counts().reset_index()
                age_groups.columns = ['Age Group', 'Count']
                fig = px.pie(age_groups, names='Age Group', values='Count', title="Diseases Ages Distribution")
                st.plotly_chart(fig)

            with col_Age2:
                code_counts = df_clean['id'].value_counts().reset_index()
                code_counts.columns = ['Patient ID', 'Count']
                fig = px.pie(code_counts, names='Count', values='Patient ID', title="ICD Code Distribution")
                st.plotly_chart(fig)

    # Tab 3: Disease Distribution
    elif st.session_state.active_tab == "Disease Distribution":
        with st.spinner("Loading Disease Distribution..."):
            col_Disea1, col_Disea2 = st.columns(2)

            with col_Disea1:
                top_all = df['ICD_code'].value_counts().nlargest(5).reset_index()
                top_all.columns = ['ICD Code', 'Count']
                fig = px.pie(top_all, names='ICD Code', values='Count', title="Top 5 Diseases (All)")
                st.plotly_chart(fig)

            with col_Disea2:
                top_clean = df_clean['ICD_code'].value_counts().nlargest(5).reset_index()
                top_clean.columns = ['ICD Code', 'Count']
                fig = px.pie(top_clean, names='ICD Code', values='Count', title="Top 5 Diseases (Clean)")
                st.plotly_chart(fig)

            st.subheader("ICD Code Analysis")
            icd_codes = df_clean['ICD_code'].dropna().value_counts().head(20).index
            selected_icd = st.selectbox("Select an ICD Code (Top 20 Diseases)", options=sorted(icd_codes))
            
            if selected_icd:
                # Filter patient IDs having the selected ICD code
                filtered_patients = df_clean[df_clean['ICD_code'] == selected_icd]['id'].unique()
                
                # Get all records for those patients
                related_records = df_clean[df_clean['id'].isin(filtered_patients)]
                
                # Create chart
                disease_counts = related_records['ICD_code'].value_counts().nlargest(10).reset_index()
                disease_counts.columns = ['ICD Code', 'Patient Count']
                
                fig = px.bar(disease_counts, x='ICD Code', y='Patient Count', 
                            title=f"Top 10 Diseases for Patients with {selected_icd}", 
                            color='ICD Code', height=600)
                fig.update_layout(xaxis=dict(tickangle=-45))
                st.plotly_chart(fig)

    # Tab 4: Disease Comorbidity Matrix
    elif st.session_state.active_tab == "Comorbidity Matrix":
        with st.spinner("Loading Comorbidity Matrix..."):
            st.title("Comorbidity Matrix")
            filter_list = df_clean['Codes'].dropna().value_counts().index[:10]  
            selected_codes = st.multiselect("Select ICD Codes", options=filter_list, default=list(filter_list[:3]))
            
            if selected_codes:
                df_filtered = df_clean[df_clean['Codes'].isin(selected_codes)]
                if not df_filtered.empty:
                    df_pivot = pd.pivot_table(df_filtered[['id', 'ICD_code']], 
                                            index="id", columns="ICD_code", 
                                            aggfunc=lambda x: 1, fill_value=0)
                    disease_counts = df_pivot.T @ df_pivot
                    fig = px.imshow(disease_counts, text_auto=True, aspect="auto", 
                                title="Comorbidity Matrix")
                    st.plotly_chart(fig)
                else:
                    st.warning("No data matches the selected filters.")

    # Tab 5: Ask Agent (Optimized)
    elif st.session_state.active_tab == "Ask Agent":
        with st.spinner("Loading Ask Agent..."):
            agent_tab(df_clean)
        
    # Tab 6: Second Dashboard
    elif st.session_state.active_tab == "Second Dashboard":
        st.markdown("<a href='http://52.14.210.82:8501' target='_blank'>Click here to go to Second Dashboard</a>", unsafe_allow_html=True)

# Main logic to switch between login and app
if __name__ == "__main__":
    #st.title("LLM-Powered Diseases Analysis")
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        # For testing, skip login
        #main_app()
        login_page()
    else:
        main_app()