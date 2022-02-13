# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:35:35 2020

@author: Yichen Jiang
"""

"""
--- read twitter raw data and process ---
--- of BIDEN ---

"""


# In[]
import os
import pandas as pd
import json
import glob
from nltk.stem.snowball import EnglishStemmer
from tqdm import tqdm 
import numpy as np
from textblob import TextBlob
import string
from datetime import datetime, timedelta


# In[]
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Debate Data\Biden\*\*\*\*'

list_path = glob.glob(path)

# In[]
"""
--- define function for quickly checking polarity_type and subjectivity_type ---

"""
def check_polarity(polarity):
    if polarity > 0.0:
        polarity_type = 'positive'
    elif polarity == 0.0:
        polarity_type = 'neutral'
    elif polarity < 0.0:
        polarity_type = 'negative'
    return polarity_type

def check_subjectivity(subjectivity):
    if subjectivity >= 0.5:
        subjectivity_type = 'subjective'
    elif subjectivity < 0.5:
        subjectivity_type = 'objective'
    return subjectivity_type

# In[]
list_category = ['hash', 'to', 'from', 'at']
list_sentiment = ['negative', 'positive', 'neutral', 'objective', 'subjective']


list_var = []
list_var.append('count')
list_var.append('count_retweet')
list_var.append('count_joe_mentioned')
list_var.append('count_biden_mentioned')
list_var.append('count_donation_mentioned')

for sentiment in list_sentiment:
    list_var.append('count_'+str(sentiment))
    if sentiment != 'neutral':
        list_var.append('ave_'+str(sentiment))
    for category in list_category:
        if 'count_'+str(category) not in list_var:
            list_var.append('count_'+str(category))
        list_var.append('count_'+str(category)+'_'+str(sentiment))
        if sentiment != 'neutral':
            list_var.append('ave_'+str(category)+'_'+str(sentiment))

list_list = ['list_negative','list_positive','list_neutral','list_objective','list_subjective']
for category in list_category:
    for sentiment in list_sentiment:
        list_list.append('list_'+str(category)+'_'+str(sentiment))

# In[]
dict_twitter = {}
stemmer = EnglishStemmer()

# In[]
""" --- find link relates to donation link --- """
""" --- from tweets sent from joebiden --- """
df_twitter = pd.DataFrame()

for link in list_path:
    if 'from_JoeBiden' in link:
        df_data = pd.read_csv(link,  error_bad_lines=False, warn_bad_lines=False)
        df_twitter = df_twitter.append(df_data)

df_twitter = df_twitter.reset_index(drop=True)

# In[]
""" --- preprocess tweets sent from biden --- """
df_temp = pd.DataFrame()
for index in df_twitter.index:
    raw_text = df_twitter.loc[index]['statusText']
    text = TextBlob(raw_text).words
    ### if related to donation
    # Stemming
    list_word = []
    for word in text:
        word = stemmer.stem(word)
        if word not in string.punctuation:
            list_word.append(word)
    if 'donat' in list_word or 'donor' in list_word:
        print(raw_text)
        df_temp = df_temp.append(pd.DataFrame(df_twitter.loc[index]).T)
        
# In[]
        
list_links = ['https://t.co/h2fvoyS16A','https://t.co/iGrbvPmRlF']

# In[]

"""
--- data processing ---
--- category recognition ---
--- sentiment analysis ---

"""
count = 0
list_record = []

for link in tqdm(list_path):
    # read data
    df_data = pd.read_csv(link,  error_bad_lines=False, warn_bad_lines=False, low_memory = False)
    if count == 0:
        list_columns = list(df_data.columns)
        count += 1
        
    # incorrect reading of df index
    if df_data.index[0] != 0:
        # save df with incorrect index reading
        list_record.append(link)
        if df_data.index[0] == 'BATCH_REFERENCE' or df_data.index[0] == 'BATCH_TIMELINE':
            # reset index
            df_data = df_data.reset_index()
            # change column name
            ## create dict_names
            dict_names = {}
            for i in range(len(list_columns)):
                dict_names[list(df_data.columns)[i]] = list_columns[i]
            # remove the redundent column
            del df_data[df_data.columns[-1]]
            # rename column
            df_data = df_data.rename(columns=dict_names)
    
    for index in df_data.index:
        
        if df_data.loc[index]['statusFilterType'] != 'BATCH_TIMELINE' and \
        df_data.loc[index]['statusFilterType'] != 'BATCH_REFERENCE':
            continue
        
        date = datetime.strftime(datetime.strptime(df_data.loc[index]['statusCreatedAt'], "%a %b %d %H:%M:%S EDT %Y"),"%Y-%m-%d")
        status_id = df_data.loc[index]['statusId']        
    
        # create dict for a new date
        if date not in dict_twitter.keys():
            dict_twitter[date] = {}
            for var in list_var:
                dict_twitter[date][var] = 0
            for l in list_list:
                dict_twitter[date][l] = []
        
        # check category
        ## count
        dict_twitter[date]['count'] += 1
        
        ## sentiment
        polarity = TextBlob(df_data.loc[index]['statusText']).sentiment.polarity
        subjectivity = TextBlob(df_data.loc[index]['statusText']).sentiment.subjectivity
        polarity_type = check_polarity(polarity)
        subjectivity_type = check_subjectivity(subjectivity)
        dict_twitter[date]['count_'+str(polarity_type)] += 1
        dict_twitter[date]['count_'+str(subjectivity_type)] += 1
        dict_twitter[date]['list_'+str(polarity_type)].append(polarity)
        dict_twitter[date]['list_'+str(subjectivity_type)].append(subjectivity)

        ## if retweet
        if df_data.loc[index]['statusIsRetweet'] == True:
            dict_twitter[date]['count_retweet'] += 1    
    
        ## if hash
        list_temp = []
        for column in df_data.columns:
            if 'Hashtag' in column: #  and type(df_data.loc[index][column]) == str
                list_temp.append(df_data.loc[index][column])
        for hashtag in list_temp:
            if type(hashtag) != str:
                continue
            if 'joe' in hashtag.lower() or 'biden' in hashtag.lower():
                dict_twitter[date]['count_hash'] += 1
                ### sentiment
                dict_twitter[date]['count_hash_'+str(polarity_type)] += 1
                dict_twitter[date]['count_hash_'+str(subjectivity_type)] += 1
                dict_twitter[date]['list_hash_'+str(polarity_type)].append(polarity)
                dict_twitter[date]['list_hash_'+str(subjectivity_type)].append(subjectivity)
                break
        
        ## if to # JoeBiden's user_id = '939091'
        if df_data.loc[index]['statusInReplyToUserId'] == 939091:
            dict_twitter[date]['count_to'] += 1
            ### sentiment
            dict_twitter[date]['count_to_'+str(polarity_type)] += 1
            dict_twitter[date]['count_to_'+str(subjectivity_type)] += 1
            dict_twitter[date]['list_to_'+str(polarity_type)].append(polarity)
            dict_twitter[date]['list_to_'+str(subjectivity_type)].append(subjectivity)
        
        ## if from
        if df_data.loc[index]['statusUserId'] == 939091:
            dict_twitter[date]['count_from'] += 1
            ### sentiment
            dict_twitter[date]['count_from_'+str(polarity_type)] += 1
            dict_twitter[date]['count_from_'+str(subjectivity_type)] += 1
            dict_twitter[date]['list_from_'+str(polarity_type)].append(polarity)
            dict_twitter[date]['list_from_'+str(subjectivity_type)].append(subjectivity)    
            ### if related to donation
            text = TextBlob(df_data.loc[index]['statusText']).words
            # Stemming
            list_word = []
            for word in text:
                word = stemmer.stem(word)
                if word not in string.punctuation:
                    list_word.append(word)
            #### if contain donation-related words: 'donat' 'donor'
            if 'donat' in list_word or 'donor' in list_word:
                dict_twitter[date]['count_donation_mentioned'] += 1    
            #### link in text
            else:
                for link in list_links:
                    if link in df_data.loc[index]['statusText']:
                        dict_twitter[date]['count_donation_mentioned'] += 1  

        ## if at
        if 'at_JoeBiden' in link:
            dict_twitter[date]['count_at'] += 1
            ### sentiment
            dict_twitter[date]['count_at_'+str(polarity_type)] += 1
            dict_twitter[date]['count_at_'+str(subjectivity_type)] += 1
            dict_twitter[date]['list_at_'+str(polarity_type)].append(polarity)
            dict_twitter[date]['list_at_'+str(subjectivity_type)].append(subjectivity)                
   
        # no of 'joe' mentioned & no of 'biden' mentioned
        ## lower case
        text = df_data.loc[index]['statusText'].lower()
        if 'joe' in text:
            dict_twitter[date]['count_joe_mentioned'] += 1
        if 'biden' in text:
            dict_twitter[date]['count_biden_mentioned'] += 1
            

  


# In[]
""" --- export dict --- """

with open(os.path.join(r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Debate Data\Biden', 'dict_biden.json'),'w+',encoding='utf-8') as file:
    json.dump(dict_twitter, file, ensure_ascii=False) 
    

# In[]
    
"""
--- data processing ---
--- metrics calculation ---

"""
for date in tqdm(dict_twitter.keys()):
    for var in dict_twitter[date].keys():
        if 'list_' in var and 'neutral' not in var:
            metric = var[5:len(var)]
            ave = np.mean(dict_twitter[date][var])
            dict_twitter[date]['ave_'+str(metric)] = ave

# In[]
list_dates = list(dict_twitter.keys())
list_dates.sort(reverse=False)

# In[]
dict_data = {}
      
for date in tqdm(list_dates):
    date_time = datetime.strptime(str(date),'%Y-%m-%d')
    if date not in dict_data.keys():
        dict_data[date] = {}
                
    ## for independent variables
    for independentvar in list_var:
        for i in range(1,4):
            # data on current day
            dict_data[date][str(independentvar)] = dict_twitter[date][independentvar]
            pastdate = (date_time-timedelta(days=i)).strftime('%Y-%m-%d')
            # if pastdate is missing
            if pastdate not in dict_twitter.keys():
                dict_data[date][str(independentvar)+'-'+str(i)] = 0
            # if pastdate exists
            else: 
                dict_data[date][str(independentvar)+'-'+str(i)] = dict_twitter[pastdate][independentvar]
                                        


# In[]
""" dict -> df """
df_data = pd.DataFrame.from_dict(dict_data).T

""" sort df by index """
df_data = df_data.sort_index(axis = 0,ascending = True)


# In[]

""" --- export df --- """

df_data.to_csv(os.path.join(r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Debate Data\Biden', 'df_twitter_biden.csv'), index=True, quoting=1)







