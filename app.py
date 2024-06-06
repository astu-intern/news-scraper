import streamlit as st
import pandas as pd
import subprocess
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from functions import category_predictor, status_predictor, predictions_compiler, webscraper
import requests
from bs4 import BeautifulSoup
import csv
import string
import datetime
import csv


# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # To read file as dataframe
    df = pd.read_csv(uploaded_file, encoding='cp1252')
    df = df.head(1)
    # Display the dataframe
    st.write(df)
    
    # Save the uploaded file to process it
    with open("uploaded_file.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File saved successfully.")

    keywords_file = 'trial_keywords.csv'
    output_filename = 'webscraper_output'

    headlines_df = webscraper(df, keywords_file, output_filename)
    st.write(headlines_df)
    

    unpredicted_data = category_predictor(headlines_df)
    predicted_data = status_predictor(unpredicted_data)
    results = predictions_compiler(predicted_data)

    for i in results:
        st.write(i)

    for k in predicted_data:
        st.write(k)