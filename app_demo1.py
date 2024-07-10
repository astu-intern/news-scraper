import streamlit as st
import pandas as pd
import subprocess
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from functions_demo1 import category_predictor, status_predictor, predictions_compiler, webscraper
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

    keywords_file = 'keywordsfile.csv'
    output_filename = 'webscraper_output'

    df['Tertiary Keywords'] = df['CITY']
    df = df[['Tertiary Keywords','START_DATE','END_DATE']]
    

    headlines_df = webscraper(df, keywords_file, output_filename)



    transport_tokenizer = "augmented_models/transport/tokenizer"
    transport_model = "augmented_models/transport/model"
    industrial_tokenizer = "augmented_models/transport/tokenizer"
    industrial_model = "augmented_models/transport/model"
    manmade_tokenizer = "augmented_models/transport/tokenizer"
    manmade_model = "augmented_models/transport/model"
    natural_tokenizer = "augmented_models/transport/tokenizer"
    natural_model = "augmented_models/transport/model"



    unpredicted_data = category_predictor(headlines_df)
    predicted_data = status_predictor(unpredicted_data[3],transport_model, transport_tokenizer)
    st.header('Predictions:')
    st.write(predicted_data)
