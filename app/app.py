import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#---------------------------------#
# Page layout
# Page expands to full width
st.set_page_config(page_title='EHR Symptom Detection',
    layout='wide')

#---------------------------------#
st.write("""
# EHR Symptom Detection
This application is built to detect symptoms in an EHR notes, including dyspnea, chest pain, fatique, nausea and cough.
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

if st.button('add'):
    result = add(1, 2)
    st.write('result: %s' % result)

#---------------------------------#
# Main panel