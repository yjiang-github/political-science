# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:12:55 2019

@author: Yichen Jiang
"""

"""
--- New Preprocessing File!!! ---
--- this file is for re-importing democraticdebate2020 data and collecting tweets for sentiment analysis ---

"""

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

# In[]
createVar = locals()

# In[]: # define plot color generator
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ''
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return '#'+color

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
count_tweet = 0
for hashtag in dict_tweets.keys():
    count_tweet += len(dict_tweets[hashtag])
    
# In[]
print('total number of tweets:', str(count_tweet))
# 1090724

# In[]
"""
--- sort data by each user ---

"""
dict_user = {}

for hashtag in dict_tweets.keys():
    for status_id in dict_tweets[hashtag].keys():
        user_id = dict_tweets[hashtag][status_id]['user_id']
        status_id = dict_tweets[hashtag][status_id]['status_id']
        # add user into dict_user
        if user_id not in dict_user.keys():
            dict_user[user_id] = {}
        if status_id not in dict_user[user_id].keys():
            dict_user[user_id][status_id] = dict_tweets[hashtag][status_id]
            dict_user[user_id][status_id]['hashtag_collectedby'] = [hashtag]
        elif status_id in dict_user[user_id].keys():
            if hashtag not in dict_user[user_id][status_id]['hashtag_collectedby']:
                dict_user[user_id][status_id]['hashtag_collectedby'].append(hashtag)

# In[]
"""
--- check number of hashtags for each tweet ---

"""
count_hashtag = 0
for user_id in dict_user.keys():
    for status_id in dict_user[user_id].keys():
        if len(dict_user[user_id][status_id]['hashtag_collectedby']) >= 2:
            count_hashtag += 1

# In[]
print('number of same tweet collected by more than 1 hashtag:', str(count_hashtag))
# 124313
# In[]
"""
--- perform sentiment analysis for each user with each of their tweet ---

"""

for user_id in dict_user.keys():
    for status_id in dict_user[user_id].keys():
        text_temp = TextBlob(dict_user[user_id][status_id]['text'])
        dict_user[user_id][status_id]['sentiment_polarity'] = text_temp.sentiment.polarity
        dict_user[user_id][status_id]['sentiment_subjectivity'] = text_temp.sentiment.subjectivity

# In[]
"""
--- find influential users with their tweets and track sentimental trend ---

"""
# set thresholds
num_followers = 500000
num_tweets = 5000
num_friends = 1000

dict_influentialuser = {}
# define time-comparing variables
latest_time = 0
curr_time = 0
latest_status_id = ''

for user_id in dict_user.keys():
    # find the lastest tweet_time this user has
    for status_id in dict_user[user_id].keys():
        curr_time = datetime.strptime(dict_user[user_id][status_id]['created_at'], "%Y-%m-%d %H:%M:%S")
        if latest_time == 0:
            latest_time = curr_time
        elif (latest_time - curr_time).days < 0:
            latest_time = curr_time
    # find the lastest tweet this user has, by tweet_time
    for status_id in dict_user[user_id].keys():
        if datetime.strptime(dict_user[user_id][status_id]['created_at'], "%Y-%m-%d %H:%M:%S") == latest_time:
            break
    
    # check thresholds
    followers_count = dict_user[user_id][status_id]['followers_count']
    statuses_count = dict_user[user_id][status_id]['statuses_count']
    friends_count = dict_user[user_id][status_id]['friends_count']
    
    if followers_count > num_followers and statuses_count > num_tweets and friends_count > num_friends:
        user_name = dict_user[user_id][status_id]['name']
        dict_influentialuser[str(user_name)+'_'+str(user_id)] = dict_user[user_id]
        print(dict_user[user_id][status_id]['name'])

# In[]
"""
--- Andrew Yang, CNN, NBC News, Donald Trump Jr.---

"""
for user in dict_influentialuser.keys():
    # get user_name
    user_name = user.split('_')[0]
    if user_name == 'Andrew Yang':
        dict_tweets_ay = dict_influentialuser[user]
    elif user_name == 'CNN':
        dict_tweets_cnn = dict_influentialuser[user]
    elif user_name == 'NBC News':
        dict_tweets_nbc = dict_influentialuser[user]
    elif user_name == 'Donald Trump Jr.':
        dict_tweets_dtj = dict_influentialuser[user]

# In[]
list_user = ['ay', 'cnn', 'nbc','dtj']
# sort data by time

for user in list_user:
    createVar['dict_sort_'+str(user)] = {}
    for status_id in createVar['dict_tweets_'+str(user)].keys():
        created_at = createVar['dict_tweets_'+str(user)][status_id]['created_at']
        time = int(created_at[5:7]+created_at[8:10]+created_at[11:13]+created_at[14:16]+created_at[17:19])
        polarity = createVar['dict_tweets_'+str(user)][status_id]['sentiment_polarity']
        subjectivity = createVar['dict_tweets_'+str(user)][status_id]['sentiment_subjectivity']
        text = createVar['dict_tweets_'+str(user)][status_id]['text']        
        
        
        
        createVar['dict_sort_'+str(user)][time] = {}
        createVar['dict_sort_'+str(user)][time]['polarity'] = polarity
        createVar['dict_sort_'+str(user)][time]['subjectivity'] = subjectivity
        createVar['dict_sort_'+str(user)][time]['text'] = text

# In[]
# sort 
for user in list_user:
    createVar['dict_plot_'+str(user)] = {'time':[],'polarity':[],'subjectivity':[],'text':[]}
    list_temp = sorted(createVar['dict_sort_'+str(user)].items(),key=lambda x:x[0],reverse=False)
    for item in list_temp:
        createVar['dict_plot_'+str(user)]['time'].append(item[0])
        createVar['dict_plot_'+str(user)]['polarity'].append(item[1]['polarity'])
        createVar['dict_plot_'+str(user)]['subjectivity'].append(item[1]['subjectivity'])
        createVar['dict_plot_'+str(user)]['text'].append(item[1]['text'])
        

# In[];
"""
--- draw plots ---

"""
dict_username = {'ay':'Andrew Yang', 'cnn':'CNN', 'nbc':'NBC NEWS', 'dtj':'Donald Trump Jr.'}

for user in list_user:
    length = len(createVar['dict_plot_'+str(user)]['time'])
    #font_size = dict_size[length]
    plt.style.use('ggplot')
    fig = plt.figure(dpi = 80, figsize = (1.25*length, 0.5*length))
    plt.title('User_name: '+str(dict_username[user]),  fontdict={'size':'30'})
    plt.subplot(1, 1, 1)

    polarity = createVar['dict_plot_'+str(user)]['polarity']
    subjectivity = createVar['dict_plot_'+str(user)]['subjectivity']
    index = np.arange(length)
    plt.plot(index, polarity, label='Polarity', color='r') 
    plt.plot(index, subjectivity, label='Subjectivity', color='k') 
    
    plt.xlabel('Date', fontdict={'size':'30'})
    plt.ylabel('Polarity & Subjectivity', fontdict={'size':'30'})
    plt.xticks(index, createVar['dict_plot_'+str(user)]['time'], fontsize=15,rotation = 45)
    #plt.xticks(index, dict_plot[candidate]['time'], fontsize=font_size, rotation = 45)
    plt.yticks(fontsize = 30)
    plt.legend(loc='upper right',fontsize = 10)
    plt.show()    















# In[]
"""
--- other code ---

"""
"""
--- save dict_tweets ---

"""
with open(os.path.join(path, 'dict_tweets.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_tweets, outfile, ensure_ascii=False)         



# In[]
"""
--- check deltatime ---

"""
# early
time1 = datetime.strptime('2019-08-01 01:40:51', "%Y-%m-%d %H:%M:%S")
#late
time2 = datetime.strptime('2019-08-01 01:40:56', "%Y-%m-%d %H:%M:%S")

delta = time1-time2

delta.days < 0

# In[]
"""
--- sort data for drawing plots ---

"""


for user in list_user:
    createVar['dict_plot_'+str(user)] = {'time':[],'polarity':[],'subjectivity':[],'text':[]}
    for status_id in createVar['dict_tweets_'+str(user)].keys():
        created_at = createVar['dict_tweets_'+str(user)][status_id]['created_at']
        
        time = int(created_at[5:7]+created_at[8:10]+created_at[11:13]+created_at[14:16]+created_at[17:19])
        polarity = createVar['dict_tweets_'+str(user)][status_id]['sentiment_polarity']
        subjectivity = createVar['dict_tweets_'+str(user)][status_id]['sentiment_subjectivity']
        text = createVar['dict_tweets_'+str(user)][status_id]['text']
        
        createVar['dict_plot_'+str(user)]['time'].append(time)
        createVar['dict_plot_'+str(user)]['polarity'].append(polarity)
        createVar['dict_plot_'+str(user)]['subjectivity'].append(subjectivity)
        createVar['dict_plot_'+str(user)]['text'].append(text)
        
