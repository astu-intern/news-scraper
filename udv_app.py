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
import numpy as np

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

    df['Tertiary Keywords'] = df ['CITY']
    df = df[['Tertiary Keywords', 'START_DATE', 'END_DATE']]
    
    headlines_df = webscraper(df, keywords_file, output_filename)
    st.header('Headlines:')
    headlines_df.index = np.arange(1, len(headlines_df) + 1)
    st.dataframe(headlines_df, use_container_width = True)






    unpredicted_data = category_predictor(headlines_df)
    #st.write(unpredicted_data)


    # List of all the models
    transport_tokenizer = "augmented_models/transport/tokenizer"
    transport_model = "augmented_models/transport/model"
    industrial_tokenizer = "augmented_models/transport/tokenizer"
    industrial_model = "augmented_models/transport/model"
    manmade_tokenizer = "augmented_models/transport/tokenizer"
    manmade_model = "augmented_models/transport/model"
    natural_tokenizer = "augmented_models/transport/tokenizer"
    natural_model = "augmented_models/transport/model"
    predicted_data = []

    transport = st.checkbox("Transport")
    if transport:
        st.header('Predictions:')
        predicted_data = status_predictor(unpredicted_data[3], transport_model,  transport_tokenizer)
        predicted_data.index = np.arange(1, len(predicted_data) + 1)
        st.dataframe(predicted_data, use_container_width = True)
        #results = predictions_compiler(predicted_data)

        #loc_pred_data = location_status_predictor(unpredicted_data)
        #loc_results = predictions_compiler(loc_pred_data)
