import streamlit as st
import pandas as pd
from data_logic import DataManager
from transformers import pipeline

# 1. Setup Page
st.set_page_config(page_title="Data Quality Checker", layout="wide")
st.title("📊 Automated Data Quality & AI Insights")

# 2. Load AI Model (Using the 'Base' model you downloaded)
@st.cache_resource
def load_ai():
    return pipeline("text2text-generation", model="google/flan-t5-base")

ai_brain = load_ai()

# 3. File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    dm = DataManager(uploaded_file)
    
    # --- SECTION 1: DATA OVERVIEW ---
    st.subheader("1. Raw Data Preview")
    st.dataframe(dm.df.head())
    
    # --- SECTION 2: ISSUE DETECTION ---
    st.subheader("2. Detected Issues")
    issues = dm.analyze_issues()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", issues['rows'])
    col2.metric("Duplicates Found", issues['duplicates'])
    col3.metric("Missing Columns", len(issues['missing']))

    if issues['missing']:
        st.warning(f"⚠️ Missing Values Detected: {list(issues['missing'].keys())}")
    else:
        st.success("✅ No missing values found!")

    # --- SECTION 3: AUTO-CLEANING ---
    st.subheader("3. Data Cleaning")
    if st.button("Clean Data Now 🧹"):
        cleaned_df = dm.clean_data()
        st.success("Duplicates removed & Missing values filled!")
        st.write("Preview of Cleaned Data:")
        st.dataframe(cleaned_df.head())
        
        # --- SECTION 4: AI INSIGHTS ---
        st.subheader("4. AI Analysis 🤖")
        with st.spinner("Generating Report..."):
            
            # We explicitly tell the AI to write a recommendation sentence
            missing_cols = ", ".join(issues['missing'].keys())
            
            input_text = (
                f"The columns {missing_cols} have missing data. "
                "Write a sentence recommending to fill these missing values with the average or 'Unknown'."
            )
            
            # We set min_length=10 to force it to write more than just one word
            insight = ai_brain(input_text, min_length=15, max_length=100)[0]['generated_text']
            
            st.info(f"**AI Recommendation:** {insight}")
            
            # Save Report
            with open("report.txt", "w") as f:
                f.write(f"DATA REPORT\nIssues: {issues}\n\nAI Insight:\n{insight}")
            st.caption("Report saved to report.txt")