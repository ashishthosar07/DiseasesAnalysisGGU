import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Patient Data Analysis Dashboard", layout="wide")

# Title
st.title("Patient Data Analysis Dashboard")

st.header("Age Filters")
min_age, max_age = st.slider("Select Age Range", 0, 100, (10, 50), step=10)

# Data loading and preprocessing
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('Patient Data Updated.csv', skipinitialspace=True)
        df = df[df['ICD_code'] != '0']
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        now = datetime.now()
        df['age'] = df['birthdate'].apply(
            lambda x: now.year - x.year - ((now.month, now.day) < (x.month, x.day)) 
            if pd.notna(x) else np.nan
        )
        df['ICD_code'] = df['ICD_code'].str.split(',')
        df = df.explode('ICD_code')
        df['ICD_code'] = df['ICD_code'].fillna('Unknown')
        df[['Codes', 'Diseases']] = df['ICD_code'].str.extract(r'\(([^)]+)\)\s*(.*)')
        df['Diseases'] = np.where(df['Codes'].isna(), df['ICD_code'], df['Diseases'])
        df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
        df_unspecified = df[df['Codes'].isna()]
        df_clean = df[~df['Codes'].isna()]
        return df, df_unspecified, df_clean
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Load data
df, df_unspecified, df_clean = load_and_process_data()
df = df[(df['age'] >= min_age) & (df['age'] <= max_age)]
df_unspecified = df_unspecified[(df_unspecified['age'] >= min_age) & (df_unspecified['age'] <= max_age)]
df_clean = df_clean[(df_clean['age'] >= min_age) & (df_clean['age'] <= max_age)]


if df is None:
    st.stop()

def get_age_group(age):
    if age < 20: return '0-20'
    elif age < 30: return '20-29'
    elif age < 50: return '30-49'
    else: return '50+'

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Age Analysis", "Disease Distribution", "Comorbidity Matrix"])

# Tab 1: Overview
with tab1:
    #filtered_df = df[(df['age'] >= min_age) & (df['age'] <= max_age)]
    filtered_df = df
    #filtered_df_clean = df_clean[(df_clean['age'] >= min_age) & (df_clean['age'] <= max_age)]
    filtered_df_clean = df_clean
    #filtered_df_unspecified = df_unspecified[(df_unspecified['age'] >= min_age) & (df_unspecified['age'] <= max_age)]
    filtered_df_unspecified = df_unspecified
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", df['id'].nunique())
    col2.metric("Total Patients Explode Count", len(filtered_df))
    col3.metric("Unspecified Diseases Count", len(filtered_df_unspecified))
    col4.metric("Clean Diseases Count", len(filtered_df_clean))
    
    col_raw, col_chart = st.columns(2)
    with col_raw:
        st.subheader("Raw Data")
        st.dataframe(filtered_df_clean[["id", "ICD_code", "age"]])
    
    with col_chart:
        st.subheader("Patient Distribution by Age Group")
        age_groups = filtered_df['age'].apply(get_age_group).value_counts().reset_index()
        age_groups.columns = ['Age Group', 'Number of Patients']
        fig = px.bar(age_groups, x='Age Group', y='Number of Patients', title="Patient Distribution by Age Group", color='Age Group')
        st.plotly_chart(fig)
    
    st.subheader("Top 20 Diseases")
    top_diseases = filtered_df_clean['ICD_code'].value_counts().nlargest(20).reset_index()
    top_diseases.columns = ['ICD Code', 'Number of Patients']
    fig = px.bar(top_diseases, x='ICD Code', y='Number of Patients', title="Top 20 Diseases by Number of Patients", color='ICD Code', height=800)
    fig.update_layout(xaxis=dict(tickangle=-45))
    st.plotly_chart(fig)

# Tab 2: Age Analysis
with tab2:
    col_Age1, col_Age2 = st.columns(2)
    with col_Age1:
        age_groups = filtered_df_clean['age'].apply(get_age_group).value_counts().reset_index()
        age_groups.columns = ['Age Group', 'Count']
        fig = px.pie(age_groups, names='Age Group', values='Count', title="Diseases Ages Distribution")
        st.plotly_chart(fig)

    with col_Age2:
        age_groups = df_clean['id'].value_counts().reset_index()
        #age_groups = df_clean.loc[(df_clean['age'] >= min_age) & (df_clean['age'] <= max_age), ["id"]].value_counts().reset_index()
        age_groups.columns = ['Patient Count', 'Count']
        #print(age_groups)
        fig = px.pie(age_groups, names='Count', values='Patient Count', title="ICD Code Distribution")
        st.plotly_chart(fig)

        
    # st.subheader("ICD Code Analysis")
    # #filtered_df_clean = df_clean[(df_clean['age'] >= min_age) & (df_clean['age'] <= max_age)]
    # filtered_df_clean = df_clean
    # icd_codes = filtered_df_clean['ICD_code'].dropna().unique()
    # selected_icd = st.selectbox("Select an ICD Code", options=sorted(icd_codes))
    
    # if selected_icd:
    #     # Step 1: Filter patient IDs having the selected ICD code
    #     filtered_patients = filtered_df_clean[filtered_df_clean['ICD_code'] == selected_icd]['id'].unique()
        
    #     # Step 2: Use these patient IDs to filter all related records from the source data
    #     related_records = filtered_df_clean[filtered_df_clean['id'].isin(filtered_patients)]
        
    #     # Step 3: Aggregate data and create a bar chart for top 10 diseases
    #     disease_counts = related_records['ICD_code'].value_counts().nlargest(10).reset_index()
    #     disease_counts.columns = ['ICD Code', 'Patient Count']
        
    #     fig = px.bar(disease_counts, x='ICD Code', y='Patient Count', 
    #                  title=f"Top 10 Diseases for Patients with {selected_icd}", 
    #                  color='ICD Code', height=600)
    #     fig.update_layout(xaxis=dict(tickangle=-45))
    #     st.plotly_chart(fig)

# Tab 3: Disease Distribution
with tab3:

    col_Disea1, col_Disea2 = st.columns(2)

    with col_Disea1:
        top_all = filtered_df['ICD_code'].value_counts().nlargest(5).reset_index()
        top_all.columns = ['ICD Code', 'Count']
        fig = px.pie(top_all, names='ICD Code', values='Count', title="Top 5 Diseases (All)")
        st.plotly_chart(fig)

    with col_Disea2:
        top_clean = filtered_df_clean['ICD_code'].value_counts().nlargest(5).reset_index()
        top_clean.columns = ['ICD Code', 'Count']
        fig = px.pie(top_clean, names='ICD Code', values='Count', title="Top 5 Diseases (Clean)")
        st.plotly_chart(fig)

    st.subheader("ICD Code Analysis")
    #filtered_df_clean = df_clean[(df_clean['age'] >= min_age) & (df_clean['age'] <= max_age)]
    filtered_df_clean = df_clean
    #icd_codes = filtered_df_clean['ICD_code'].dropna().unique()
    icd_codes = filtered_df_clean['ICD_code'].dropna().value_counts().head(20).index
    #icd_codes = filtered_df_clean['ICD_code'].value_counts().nlargest(20)
    #print(icd_codes)
    selected_icd = st.selectbox("Select an ICD Code (Top 20 Diseases)", options=sorted(icd_codes))
    
    if selected_icd:
        # Step 1: Filter patient IDs having the selected ICD code
        filtered_patients = filtered_df_clean[filtered_df_clean['ICD_code'] == selected_icd]['id'].unique()
        
        # Step 2: Use these patient IDs to filter all related records from the source data
        related_records = filtered_df_clean[filtered_df_clean['id'].isin(filtered_patients)]
        
        # Step 3: Aggregate data and create a bar chart for top 10 diseases
        disease_counts = related_records['ICD_code'].value_counts().nlargest(10).reset_index()
        disease_counts.columns = ['ICD Code', 'Patient Count']
        
        fig = px.bar(disease_counts, x='ICD Code', y='Patient Count', 
                     title=f"Top 10 Diseases for Patients with {selected_icd}", 
                     color='ICD Code', height=600)
        fig.update_layout(xaxis=dict(tickangle=-45))
        st.plotly_chart(fig)

    

# Tab 4: Disease Comorbidity Matrix
with tab4:
    st.title("Comorbidity Matrix")
    #filter_list = ['E11.9', 'E78.5', 'E66.9', 'E78.2', 'R73.03', 'K21.9', 'M54.50', 'E78.49', 'E03.9']
    filter_list = filtered_df_clean['Codes'].dropna().value_counts().head(10).index
    selected_codes = st.multiselect("Select ICD Codes", options=filter_list, default=filter_list[:3])
    
    if selected_codes:
        #df_filtered = df_clean[(df_clean['Codes'].isin(selected_codes)) & (df_clean['age'] >= min_age) & (df_clean['age'] <= max_age)]
        df_filtered = df_clean[(df_clean['Codes'].isin(selected_codes))]
        if not df_filtered.empty:
            df_pivot = pd.pivot_table(df_filtered[['id', 'ICD_code']], index="id", columns="ICD_code", aggfunc=lambda x: 1, fill_value=0)
            disease_counts = df_pivot.T @ df_pivot
            fig = px.imshow(disease_counts, text_auto=True, aspect="auto", title="Comorbidity Matrix")
            st.plotly_chart(fig)
        else:
            st.warning("No data matches the selected filters.")
