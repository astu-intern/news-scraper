 # all required functions
import pandas as pd
import os

import streamlit
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
from bs4 import BeautifulSoup
import csv
import string
import datetime
import csv
import streamlit as st
#import geotext
#import locationtagger
import time

# function 1 :


@st.cache_data
def status_predictor(unpredicted_data, model, tokenizer):


    # preprocessing the data
    def preprocess_text(text):
        inputs = tokenizer(text, return_tensors = 'pt', max_length = 128, padding = "max_length", truncation = True)
        return inputs['input_ids'], inputs['attention_mask']

    # map the labels
    label_to_category = {0: 0, 1: 1}


    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    model = BertForSequenceClassification.from_pretrained(model)
    model.eval()

    headlines_data = unpredicted_data
    #st.write(headlines_data)
    # apply preprocessing
    headlines_data['input_ids'], headlines_data['attention_mask'] = zip(*headlines_data['headline'].apply(preprocess_text))

    # make the predictions
    predictions = []
    for input_ids, attention_mask in zip(headlines_data['input_ids'], headlines_data['attention_mask']):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim = 1).item()
            predicted_category = label_to_category[predicted_class]
            predictions.append(predicted_category)

    # add the predictions to the dataframe
    headlines_data['predicted_status'] = predictions
    headlines_data = headlines_data[['headline','predicted_status']]

    return headlines_data

# location fn

def location_status_predictor(unpredicted_data):

    predicted_data = []
    # store all the models and tokenizers
    transport_tokenizer = "augmented_models/transport/tokenizer"
    transport_model  = "augmented_models/transport/model"
    industrial_tokenizer = "augmented_models/transport/tokenizer"
    industrial_model  = "augmented_models/transport/model"
    manmade_tokenizer = "augmented_models/transport/tokenizer"
    manmade_model  = "augmented_models/transport/model"
    natural_tokenizer = "augmented_models/transport/tokenizer"
    natural_model  = "augmented_models/transport/model"
    # NOTE: change before using

    models = [industrial_model, manmade_model, natural_model, transport_model]
    tokenizers = [industrial_tokenizer, manmade_tokenizer, natural_tokenizer, transport_tokenizer]

    # preprocessing the data
    def preprocess_text(text):
        inputs = tokenizer(text, return_tensors = 'pt', max_length = 128, padding = "max_length", truncation = True)
        return inputs['input_ids'], inputs['attention_mask']

    


    # map the labels
    label_to_category = {0: 0, 1: 1}

    for i in range(len(unpredicted_data)):
        unpredicted_data[i]["location"] = unpredicted_data[i]["headline"].apply(lambda x: location_tagger(x))

        unpredicted_data[i]['stripped_headline'] = unpredicted_data[i].apply(lambda row: strip_location(row['headline'], row['location']), axis=1)
        tokenizer = BertTokenizer.from_pretrained(tokenizers[i])
        model = BertForSequenceClassification.from_pretrained(models[i])
        model.eval()

        headlines_data = unpredicted_data[i]

        # apply preprocessing
        headlines_data['input_ids'], headlines_data['attention_mask'] = zip(*headlines_data['stripped_headline'].apply(preprocess_text))

        # make the predictions
        predictions = []
        for input_ids, attention_mask in zip(headlines_data['input_ids'], headlines_data['attention_mask']):
            with torch.no_grad():
                outputs = model(input_ids, attention_mask = attention_mask)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim = 1).item()
                predicted_category = label_to_category[predicted_class]
                predictions.append(predicted_category)

        # add the predictions to the dataframe
        headlines_data['predicted_status'] = predictions
        headlines_data = headlines_data[['headline','stripped_headline','predicted_status']]
        predicted_data.append(headlines_data)

    return predicted_data

def location_tagger(headline):
    import nltk
    import locationtagger
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

    # extracting entities.
    place_entity = locationtagger.find_locations(text = headline)
    loc_list = place_entity.countries + place_entity.regions + place_entity.cities
    return loc_list

def strip_location(headline, location_list):
  location_list = [loc.lower() for loc in location_list]
  words = headline.split()
  stripped_words = [word for word in words if word.lower() not in location_list]
  stripped_headline = ' '.join(stripped_words)
  return stripped_headline


# function 2 :

def category_predictor(headlines_data):
    # path for model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('category_model/tokenizer')
    model = BertForSequenceClassification.from_pretrained('category_model/model')
    model.eval()

    # preprocessing the data
    def preprocess_text(text):
        inputs = tokenizer(text, return_tensors = 'pt', max_length = 128, padding = "max_length", truncation = True)
        return inputs['input_ids'], inputs['attention_mask']

    # apply preprocessing
    headlines_data['input_ids'], headlines_data['attention_mask'] = zip(*headlines_data['headline'].apply(preprocess_text))

    # map the labels
    label_to_category = {0: 'industrial', 1: 'manmade', 2: 'natural', 3: 'transport'}

    # make the predictions
    predictions = []
    for input_ids, attention_mask in zip(headlines_data['input_ids'], headlines_data['attention_mask']):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim = 1).item()
            predicted_category = label_to_category[predicted_class]
            predictions.append(predicted_category)

    # add the predictions to the dataframe
    headlines_data['predicted_category'] = predictions
    headlines_data = headlines_data[['headline','predicted_category','source','city','link']]

    # sort according to category
    transport = headlines_data[headlines_data['predicted_category'] == 'transport']
    manmade = headlines_data[headlines_data['predicted_category'] == 'manmade']
    natural = headlines_data[headlines_data['predicted_category'] == 'natural']
    industrial = headlines_data[headlines_data['predicted_category'] == 'industrial']

    # save into a new csv
    transport.to_csv('data/category_wise_datasets/transport_unpredicted.csv', index = False)
    manmade.to_csv('data/category_wise_datasets/manmade_unpredicted.csv', index = False)
    natural.to_csv('data/category_wise_datasets/natural_unpredicted.csv', index = False)
    industrial.to_csv('data/category_wise_datasets/industrial_unpredicted.csv', index = False)

    return (industrial, manmade, natural, transport)


# function 4 : compile predictions

def predictions_compiler(predicted_data):
    result=[]
    k=0

    for i in predicted_data:
        total_rows = int(len(i))
        count_ones = i['predicted_status'].sum()
        percentage_ones = (count_ones / total_rows) * 100
        label_to_category = {0: 'industrial', 1: 'manmade', 2: 'natural', 3: 'transport'}

        # Print the result
        temp = "Risk percentage for "+ label_to_category[k]+" : "+ str(percentage_ones) +"%"
        result.append(temp)
        k+=1
    return result


# webscraper functions



 # @title Fxn : area_from_pincode
def area_from_pincode(pincode):
    '''
    Gets Area name from pincode
    '''
    url = "https://india-pincode-with-latitude-and-longitude.p.rapidapi.com/api/v1/pincode/" + pincode
    headers = {
        "X-RapidAPI-Key": "279e98b582mshf9cbd1afb09a97cp1ad9fbjsn87dacfde03c8",
        "X-RapidAPI-Host": "india-pincode-with-latitude-and-longitude.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    area_names = list()
    for i in response.json():
        area_names.append(i['area'])

    return area_names



# @title Fxn : keywords_from_file
def keywords_from_file(cities_df, keywords_file):
    '''
    Reads keywords from file
    '''

    cities_names = cities_df['Tertiary Keywords']
    start_date = cities_df['START_DATE']
    end_date = cities_df['END_DATE']

    query_data = pd.DataFrame()
    for i in range(len(cities_names)):
        
        temp_df = pd.read_csv(keywords_file)
        temp_df['Tertiary Keywords'] = cities_names[i]
        temp_df['START_DATE'] = start_date[i]
        temp_df['END_DATE'] = end_date[i]

        query_data = pd.concat([query_data, temp_df], ignore_index=True)

    query_data['count'] = 0
    query_data['link is present'] = False
    cols=['Primary Keywords','Secondary Keywords','Tertiary Keywords']
    query_data['keywords'] = query_data[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    query_no = query_data['Tertiary Keywords'].count()
    key_list = list(query_data['keywords'])
    st.header('Query Data:')
    st.write(query_data)
    return (query_data, query_no, key_list)



# @title Fxn : search_url
def search_url(url_base,keywords,date_filter):
    '''
    Designs the url links' list with our search keywords list
    '''
    keywords = keywords.split()
    url_list = list()

    for i in range(0,1):
      url = url_base
      url += '+'.join(keywords)
      url += "&tbm=nws&tbs=cdr:1"
      if date_filter!=0:
        url += ',cd_min:' + str(date_filter[0]) + ',cd_max:' + str(date_filter[1])+'&start='+str(i*10)
      url_list.append(url)

    #st.write(url_list)

    return url_list


# @title Fxn : GNewsWebScraper
def GNewsWebScraper(url_base,keywords,start_date_str='', end_date_str=''):
    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
    }

    if str(start_date_str) != "nan" or str(end_date_str) != "nan":
        date_filter = [start_date_str, end_date_str]
    else:
        date_filter = 0
    url_list = search_url(url_base, keywords, date_filter)
    news_results = list()
    for url in url_list:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        for el in soup.select("div.SoaBEf"):
            result = {
                "link": el.find("a")["href"],
                "headline": el.select_one("div.MBeuO").get_text(),
                "date": el.select_one(".LfVVr").get_text(),
                "source": el.select_one(".NUnG9d span").get_text()
            }
            news_results.append(result)


    return news_results


# @title Fxn : CSV_Dumper
def CSV_dumper(cities_df, url_base, query_data, query_no, key_list, output_filename, start_list=list(), end_list=list(), pincode_search=0, whitelist=0, publisher_whitelist=list()):
    try:
        os.makedirs(output_filename)
    except FileExistsError:
        # print('Folder already exists!')
        pass

    response_df = pd.DataFrame()
    area_names = list()
    for i in range(query_no):
        data_dict = {'headline': list(), 'link': list(), 'source':list(),'city':list()}
        if pincode_search == 1:
            area_names.append(area_from_pincode(str(query_data.iloc[i]['pincode'])))
        else:
            area_names.append(" ")

        for j in range(len(area_names[i])):
            if len(start_list) != 0 and len(end_list) != 0:
                data_iter = GNewsWebScraper(url_base, area_names[i][j] + '+' + key_list[i], start_list[i], end_list[i])
            else:
                data_iter = GNewsWebScraper(url_base, area_names[i][j] + '+' + key_list[i])
            st.write(data_iter)
            for k in range(len(data_iter)):
                if (not whitelist == 1) or (data_iter[k]["source"] in publisher_whitelist):
                    data_dict['headline'].append(data_iter[k]['headline'])
                    data_dict['link'].append(data_iter[k]['link'])
                    data_dict['source'].append(data_iter[k]['source'])
                    data_dict['city'].append(query_data[i]['Tertiary Keywords'])
                    try:
                        if data_iter[k]['link'] in list(query_data['Website Link']):
                            query_data.iloc[i, 8] = True
                    except:
                        pass

        df = pd.DataFrame(data_dict)
        response_df = pd.concat([response_df, df], ignore_index=True)

        csv_filename = key_list[i] + '.csv'
        
    final_df = pd.DataFrame(response_df)
    # final_df.to_csv('headlines_' + cities_file)

    return final_df


 # @title Fxn : webscraper
@st.cache_data
def webscraper(cities_df, keywords_file = 'trial_keywords.csv', output_filename='webscraper_output', pincode_search=0, whitelist=0, whitelist_file=''):
    # Importing all the relevant libraries
    #st.write('cities_df')
    #st.write(cities_df)
    if whitelist == 1:
        publisher_whitelist = list(pd.read_csv(whitelist_file)['name'])
    else:
        publisher_whitelist = list()

    url_base = 'https://www.google.com/search?q='

    # cities = (cities_file.iloc[:,0])
    query_data, query_no, key_list = keywords_from_file(cities_df, keywords_file)

    start_list = query_data['START_DATE']
    end_list = query_data['END_DATE']

    df = CSV_dumper(cities_df=cities_df, url_base=url_base, query_data=query_data, query_no=query_no, key_list=key_list,
                    output_filename=output_filename, start_list=start_list, end_list=end_list)
    st.write(df)
    return df[['headline','link','source']]












