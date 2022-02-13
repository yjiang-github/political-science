# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:17:07 2019

@author: Yichen Jiang
"""

"""
--- this file is for data cleaning and combining all data ---

"""

# In[]

import os
import json
from textblob import TextBlob
import numpy as np
from nltk.stem.snowball import EnglishStemmer
import nltk
import string
from tqdm import tqdm
import math
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import random
import csv

# In[]
createVar = locals()

# In[]:
"""
--- Define path ---

"""

path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science'

path_debate = os.path.join(path, 'Debate Data')

# In[]
"""
--- import data ---

"""
# define list of hashtags (those are the hashtags corresponding to different data files)
list_hashtags = ['DemDebate', 'democraticDebate', 'democraticdebate2020']

filenames = os.listdir(path_debate)

# In[]
"""
--- define dict_candidates ---

"""
dict_candidates = {'biden':'joe','buttigieg':'pete','harris':'kamala','sanders':'bernie','warren':'elizabeth','yang':'andrew'}

# In[]
"""
--- reading data file by each tweet ---

"""
list_temp = []

dict_tweets = {}
for filename in filenames:
    if not filename.endswith('.json'):
        continue
    for hashtag in list_hashtags:
        if hashtag in filename:
            break
    if hashtag not in dict_tweets.keys():
        dict_tweets[hashtag] = {}
    
    file = open(os.path.join(path_debate, filename),'r',encoding='utf-8')
    for line in file.readlines():
        list_temp = json.loads(line)
        for tweet in list_temp:
            status_id = tweet['status_id']
            dict_tweets[hashtag][status_id] = tweet

# In[]
"""
--- total number of tweets ---

"""

# 1090724

# In[]
"""
--- sort data by each date ---

"""

dict_date = {}

for hashtag in dict_tweets.keys():
    for status_id in dict_tweets[hashtag].keys():
        tweet_date = dict_tweets[hashtag][status_id]['created_at'][0:10]
        # add date into dict_date
        if tweet_date not in dict_date.keys():
            dict_date[tweet_date] = {}
        if status_id not in dict_date[tweet_date].keys():
            dict_date[tweet_date][status_id] = dict_tweets[hashtag][status_id]

# In[]
"""
--- extract tweets and tweet sentiment for each candidates ---

"""

dict_sentiment = {}
for candidate in dict_candidates.keys():
    dict_sentiment[candidate] = {}
    for date in dict_date.keys():
        dict_sentiment[candidate][date] = {}

for date in tqdm(dict_date.keys()):
    for status_id in dict_date[date].keys():
        text = dict_date[date][status_id]['text'].lower()
        for lastname in dict_candidates.keys():
            if lastname in text or dict_candidates[lastname] in text:
                dict_sentiment[lastname][date][status_id] = {
                        'sentiment_polarity': TextBlob(text).sentiment.polarity,
                        'sentiment_subjectivity': TextBlob(text).sentiment.subjectivity}

# In[]
"""
--- calculate mean and variance for each candidate by each day---

"""
dict_stats = {}
list_temp_pol = []
list_temp_sub = []

for candidate in dict_sentiment.keys():
    for date in dict_sentiment[candidate].keys():
        if date not in dict_stats.keys():
            dict_stats[date] = {}
        if dict_sentiment[candidate][date] == {}:
            dict_stats[date]['mean_polarity_'+candidate] = 0.0
            dict_stats[date]['mean_subjectivity_'+candidate] = 0.0
            dict_stats[date]['var_polarity_'+candidate] = 0.0
            dict_stats[date]['var_subjectivity_'+candidate] = 0.0
            dict_stats[date]['num_tweets_'+candidate] = 0
            continue
        for status_id in dict_sentiment[candidate][date].keys():
            list_temp_pol.append(dict_sentiment[candidate][date][status_id]['sentiment_polarity'])
            list_temp_sub.append(dict_sentiment[candidate][date][status_id]['sentiment_subjectivity'])
        dict_stats[date]['mean_polarity_'+candidate] = np.mean(list_temp_pol)
        dict_stats[date]['mean_subjectivity_'+candidate] = np.mean(list_temp_sub)
        dict_stats[date]['var_polarity_'+candidate] = np.var(list_temp_pol)
        dict_stats[date]['var_subjectivity_'+candidate] = np.var(list_temp_sub)
        dict_stats[date]['num_tweets_'+candidate] = len(dict_sentiment[candidate][date])
        
        list_temp_pol = []
        list_temp_sub = []
        
# In[]
"""
--- import events data ---

"""
# convert index of events data into date type (year-month-day)
def index_to_date(index):
    month, day, year = index.split('/')[0],index.split('/')[1],index.split('/')[2]
    if len(month) == 1:
        month = '0'+month
    if len(day) == 1:
        day = '0'+day
    date = year+'-'+month+'-'+day
    return date

path_events = os.path.join(path,'Events','20190630_20190807')

filenames_events = os.listdir(path_events)

# add variable into dict_stats
for filename in filenames_events:
    if filename.endswith('.csv'):
        with open(os.path.join(path_events,filename)) as file:
            name = filename.split('.')[0].split('_')[1].lower()
            for date in dict_stats.keys():
                dict_stats[date]['events_'+str(name)] = 0

for filename in filenames_events:
    if filename.endswith('.csv'):
        with open(os.path.join(path_events,filename)) as file:
            name = filename.split('.')[0].split('_')[1].lower()
            df_events = pd.read_csv(file,index_col=0)
            for index in df_events.index:
                date = index_to_date(index)
                dict_stats[date]['events_'+str(name)] += 1

# In[]
"""
--- import FEC data ---

"""
path_fec = os.path.join(path,'Fundraising data')


filenames_fec = os.listdir(path_fec)

dict_fecraw = {}

# In[]
for filename in tqdm(filenames_fec):
    if filename.endswith('.csv'):
        for candidate in dict_candidates.keys():
            if candidate in filename:
                print(candidate)
                print(filename,'\n')
                if candidate not in dict_fecraw.keys():
                    dict_fecraw[candidate] = {}
                break
        # list for saving column names
        list_columnname = []
        # line count
        count = 0
        
        file = open(os.path.join(path_fec, filename),'r',encoding='utf-8')
        lines = csv.reader(file)
        
        for line in tqdm(lines):
            # if column name
            if count == 0:
                list_columnname = line
                index_contribution_receipt_date = list_columnname.index('contribution_receipt_date')
                index_transaction_id = list_columnname.index('transaction_id')
                # not the 1st line
                count += 1
            else:
                contribution_receipt_date = line[index_contribution_receipt_date]
                transaction_id = line[index_transaction_id]
                year = contribution_receipt_date[0:4]
                month = contribution_receipt_date[5:7]
                day = contribution_receipt_date[8:10]      
                date = year+'-'+month+'-'+day
                if date not in dict_fecraw[candidate].keys():
                    dict_fecraw[candidate][date] = {}
                dict_fecraw[candidate][date][transaction_id] = {}
                for i in range(len(line)):
                    column = list_columnname[i]
                    dict_fecraw[candidate][date][transaction_id][column] = line[i]
        
# In[]
"""
--- sort fec data by each donor (name_zip) ---

"""
dict_donor = {}

for candidate in tqdm(dict_fecraw.keys()):
    if candidate not in dict_donor.keys():
        dict_donor[candidate] = {}
    for date in dict_fecraw[candidate].keys():
        for transaction_id in dict_fecraw[candidate][date].keys():
            contributor_name = dict_fecraw[candidate][date][transaction_id]['contributor_name']
            contributor_zip = dict_fecraw[candidate][date][transaction_id]['contributor_zip']
            name_zip = contributor_name+'_'+contributor_zip
            if name_zip not in dict_donor[candidate].keys():
                dict_donor[candidate][name_zip] = {}
            dict_donor[candidate][name_zip][transaction_id] = dict_fecraw[candidate][date][transaction_id]

# In[]
"""
--- check new donors per day & new donations per day ---

"""
new_donor = True

dict_fec = {}

dict_temp = {}

for candidate in tqdm(dict_donor):
    if candidate not in dict_fec.keys():
        dict_fec[candidate] = {}
    for name_zip in dict_donor[candidate].keys():
        # temp store the transaction date
        for transaction_id in dict_donor[candidate][name_zip].keys():
            contribution_receipt_date = \
            dict_donor[candidate][name_zip][transaction_id]['contribution_receipt_date'][0:10]
            year = contribution_receipt_date[0:4]
            month = contribution_receipt_date[5:7]
            day = contribution_receipt_date[8:10]
            date = year+'-'+month+'-'+day
            date_num = int(year+month+day)
            dict_temp[date_num] = contribution_receipt_date
            # save donation amount and donor number into dict_fec
            if date not in dict_fec[candidate].keys():
                dict_fec[candidate][date] = {'newdonor_num':0, 'newdonation_amount':0.0, 'donor_num':0, 'donation_amount':0.0}
            dict_fec[candidate][date]['donor_num'] += 1
            dict_fec[candidate][date]['donation_amount'] += float(dict_donor[candidate][name_zip][transaction_id]['contribution_receipt_amount'])
        # find the earliest date for current donor at current candidate
        earlist_date_num = np.array(list(dict_temp.keys())).min()
        earlist_date = dict_temp[earlist_date_num]
        
        # add new donation amount and new donor number into dict_fec
        year = earlist_date[0:4]
        month = earlist_date[5:7]
        day = earlist_date[8:10] 
        date = year+'-'+month+'-'+day
        if date not in dict_fec[candidate].keys():
            dict_fec[candidate][date] = {'newdonor_num':0, 'newdonation_amount':0.0, 'donor_num':0, 'donation_amount':0.0}
        for transaction_id in dict_donor[candidate][name_zip].keys():
            if dict_donor[candidate][name_zip][transaction_id]['contribution_receipt_date'][0:10] == earlist_date:
                dict_fec[candidate][date]['newdonor_num'] += 1
                dict_fec[candidate][date]['newdonation_amount'] += float(dict_donor[candidate][name_zip][transaction_id]['contribution_receipt_amount'])
        # clear dict_temp
        dict_temp = {}
        
# In[]
"""
--- export dict_fec into .json file ---

"""
with open(os.path.join(path, 'dict_fec.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_fec, outfile, ensure_ascii=False)    

# In[]
""" 
--- dictionary -> dataframe ---

"""
dict_fec_df = {}

for candidate in dict_fec.keys():
    if candidate not in dict_fec_df.keys():
        dict_fec_df[candidate] = pd.DataFrame.from_dict(dict_fec[candidate]).T
        dict_fec_df[candidate].sort_index(inplace=True)

# I checked Biden's raw data: there's no transaction made on 2019-04-19

# In[]
"""
--- create dictionary for saving the thresholds of numbers ---
--- create dictionary for number-month conversion ---
--- create dictionary for campaign start dates ---
--- create list for debate dates ---

"""
dict_thresholds = {'biden':{'donation_amount': 250000,
                            'donor_num': 1000,
                            'newdonation_amount': 250000,
                            'newdonor_num': 250},
                    'buttigieg':{'donation_amount': 200000,
                            'donor_num': 2000,
                            'newdonation_amount': 200000,
                            'newdonor_num': 500},
                    'harris':{'donation_amount': 200000,
                            'donor_num': 1500,
                            'newdonation_amount': 200000,
                            'newdonor_num': 400},
                    'sanders':{'donation_amount': 200000,
                            'donor_num': 5000,
                            'newdonation_amount': 100000,
                            'newdonor_num': 1000},
                    'warren':{'donation_amount': 150000,
                            'donor_num': 2500,
                            'newdonation_amount': 80000,
                            'newdonor_num': 400},
                    'yang':{'donation_amount': 50000,
                            'donor_num': 500,
                            'newdonation_amount': 20000,
                            'newdonor_num': 200}}

dict_month = {'01':'January',
              '02':'Feburary',
              '03':'March',
              '04':'April',
              '05':'May',
              '06':'June',
              '07':'July',
              '08':'Augest',
              '09':'September',
              '10':'October',
              '11':'November',
              '12':'December'}     

dict_start = {'biden':'2019-04-25',
              'buttigieg':'2019-04-14',
              'harris':'2019-01-21',
              'sanders':'2019-02-19',
              'warren':'2019-02-19',
              'yang':'2019-01-01'} # actually yang started much more eailer than others, Feb 2018

list_dates = ['2019-06-26',
              '2019-06-27',
              '2019-07-30',
              '2019-07-31',
              '2019-09-12',
              '2019-10-15',
              '2019-11-20',
              '2019-12-19',
              '2020-01-14',
              '2020-02-07',
              '2020-02-19',
              '2020-02-25']

# In[]
"""
--- plot ---
--- by candidates ---

"""

for candidate in dict_fec_df.keys():
    plt.style.use('ggplot')
     
    # plot by each candidate's each column
    for column in dict_fec_df[candidate].columns:
        
        # determine figure size
        fig = plt.figure(dpi = 80, figsize = (math.ceil((len(dict_fec_df[candidate])/10)*3), 25))
        plt.suptitle(candidate.upper()+'|'+column,  fontsize=50)
        plt.title(column+' >= '+str(dict_thresholds[candidate][column])+' has been marked',  fontdict={'size':'40'})
        
        index = np.arange(len(dict_fec_df[candidate]))
        column_name, = plt.plot(index, dict_fec_df[candidate][column],label=column,color='k')
        
        plt.xlabel('Collecting time', fontdict={'size':'50'})
        plt.ylabel(column, fontdict={'size':'50'})
        
        # create list for xticks, list for peak numbers, and setup initial month
        list_x = []
        list_marker = []
        list_marker_time = []
        month = '00'
        for num in range(len(dict_fec_df[candidate][column])):
            # if peak number has been detected
            if dict_fec_df[candidate][column].iloc[num] >= dict_thresholds[candidate][column]:
                # record the date, the number and the date number
                list_x.append(dict_fec_df[candidate][column].iloc[[num]].index.values[0][5:10])
                list_marker.append(int(dict_fec_df[candidate][column].iloc[num]))
                list_marker_time.append(num)
            else: 
                list_x.append('')
                #list_marker.append('')
            # overwrite list_x[num] if a new month has started
            if dict_fec_df[candidate][column].iloc[[num]].index.values[0][5:7] != month:
                # extract month
                month = dict_fec_df[candidate][column].iloc[[num]].index.values[0][5:7]
                list_x[num] = dict_month[dict_fec_df[candidate][column].iloc[[num]].index.values[0][5:7]]
            # overwrite list_x[num] if it is on debate dates, and, add vertical line for debate dates
            if dict_fec_df[candidate][column].iloc[[num]].index.values[0] in list_dates:
                list_x[num] = dict_fec_df[candidate][column].iloc[[num]].index.values[0][5:10]
                debate_dates = plt.axvline(num, color='red', linestyle='--', label = 'Debate Dates')
            # add vertical line for start date
            if dict_fec_df[candidate][column].iloc[[num]].index.values[0] == dict_start[candidate]:
                start_date = plt.axvline(num, color='black', linestyle='--', label = 'Start Date')
        
        # plot marker
        for num in range(len(list_marker)):
            marker = list_marker[num]
            marker_time = list_marker_time[num]
            plt.text(marker_time, marker, str(marker), fontdict={'size':'25','color':'b'})       
            
        plt.xticks(index, list_x,fontsize = 16,rotation=45)
        plt.yticks(fontsize = 50)
        plt.legend(handles = [column_name,debate_dates,start_date],loc='upper right',fontsize = 50)
        plt.savefig(os.path.join(path_fec,'plots', candidate.upper()+'_'+column+'.png'))
        plt.show()
        
    

# In[]
"""
--- plot ---
--- by categories ---

"""







# In[]
"""
--- add dict_fec into dict_stats ---

"""
for candidate in dict_fec.keys():
    for date in dict_stats.keys():
        dict_stats[date]['newdonor_num_'+candidate] = dict_fec[candidate][date]['newdonor_num']
        dict_stats[date]['newdonation_amount_'+candidate] = dict_fec[candidate][date]['newdonation_amount']
        dict_stats[date]['donor_num_'+candidate] = dict_fec[candidate][date]['donor_num']
        dict_stats[date]['donation_amount_'+candidate] = dict_fec[candidate][date]['donation_amount']

# In[]
"""
--- convert dict -> df ---

"""
df_stats = pd.DataFrame.from_dict(dict_stats).T

# In[]
"""
--- export dict_stats & df_stats---

"""

with open(os.path.join(path, 'dict_stats.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_stats, outfile, ensure_ascii=False)         

df_stats.to_csv(os.path.join(path,'df_stats.csv'), index=True, quoting=1)

# In[]
"""
--- import df_stats & dict_stats ---

"""

with open(os.path.join(path,'df_stats.csv')) as file:
    df_stats = pd.read_csv(file,header=0,index_col=0)

with open(os.path.join(path,'dict_stats.json')) as file:
    for line in file:
        dict_stats = json.loads(line)

# In[]














# In[]
"""
--- backup code ---

"""

# In[]
"""
--- sentiment analysis ---

"""
for tweet_date in tqdm(dict_date.keys()):
    for status_id in dict_date[tweet_date].keys():
        text = TextBlob(dict_date[tweet_date][status_id]['text'])
        dict_date[tweet_date][status_id]['sentiment_polarity'] = text.sentiment.polarity
        dict_date[tweet_date][status_id]['sentiment_subjectivity'] = text.sentiment.subjectivity

# In[]

for date in tqdm(dict_bernie.keys()):
    dict_fec[date] = {'newdonor_num':0, 'newdonation_amount':0.0, 'donor_num':0, 'donation_amount':0.0}
    for transaction_id in dict_bernie[date].keys():
        contributor_name = dict_bernie[date][transaction_id]['contributor_name']
        contributor_zip = dict_bernie[date][transaction_id]['contributor_zip']
        name_zip = contributor_name+'_'+contributor_zip
        for donor_transaction_id in dict_donor[name_zip].keys():
            contribution_receipt_date = dict_donor[name_zip][donor_transaction_id]['contribution_receipt_date']
            year = contribution_receipt_date
            month = contribution_receipt_date[5:7]
            day = contribution_receipt_date[8:10]      
            donation_date = year+'-'+month+'-'+day
            if donation_date < date: # current donor has donated before
                new_donor = False
            else: continue
        if new_donor == True:
            dict_fec[date]['newdonor_num'] += 1
            dict_fec[date]['newdonation_amount'] += float(dict_bernie[date][transaction_id]['contribution_receipt_amount'])
        dict_fec[date]['donor_num'] += 1
        dict_fec[date]['donation_amount'] += float(dict_bernie[date][transaction_id]['contribution_receipt_amount'])
        new_donor = True
