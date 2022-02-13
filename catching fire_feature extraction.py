# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:25:15 2020

@author: Yichen Jiang
"""

"""
--- catching fire -- feature extraction ---

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
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random
import csv
from itertools import groupby

# In[]
""" --- define path --- """
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science'
path_data = os.path.join(path,'Daily Twitter data','master_topics.csv')


# In[]


df_data = pd.read_csv(path_data,index_col = 0)


# In[]

tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
stemmer = EnglishStemmer()

# read stopword list
list_stopwords = []
path_list = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'
for line in open(os.path.join(path_list, 'Stopwords.txt'),"r"):
    list_stopwords.append(line)

SW = []
for i in range(len(list_stopwords)):
    if i % 2 == 0:
        SW.append(list_stopwords[i].replace('\n',''))

list_stopwords.clear()
list_stopwords = SW

del SW

# stopwords stemming
for i in range(0,len(list_stopwords)):
    list_stopwords[i] = stemmer.stem(list_stopwords[i])
    
# In[]
""" extract features """

list_temp = []

dict_text = {}

for i in tqdm(df_data.index.values):

    # Tokenization
    text = TextBlob(df_data.iloc[i]['text']).words
    # Stemming
    for word in text:
        word = stemmer.stem(word)
        # Punctuation removal
        word_clean = ""
                
        for elem in word:
            # if num in word then pass this word
            if elem.isdecimal():
                # word_clean = "NUM"
                # list_temp.append(word_clean)
                break
            # no num in word
            if elem.isalpha():
                word_clean += elem
            # punctuation in word: punctuation will be the splitter of the word phrase
            elif not elem.isalpha():
                # append the previous word if not in stopword list
                if word_clean not in list_stopwords and word_clean != "":
                    list_temp.append(word_clean)
                # clean the word_clean
                word_clean = ""
                
        
        # Stopwords removal
        if word_clean not in list_stopwords and word_clean != "":
            list_temp.append(word_clean)
    
    
    dict_text[i] = {'text_preprocessed': list_temp,
             'textclean_length': len(list_temp),
             'text_tokenization': text,
             'text_length': len(text),
             'text': df_data.iloc[i]['text']}
    
    list_temp = []


# In[]
""" --- total term frequency --- """
dict_wordcount = {}
for i in dict_text.keys():
    for word in dict_text[i]['text_preprocessed']:
        if word not in dict_wordcount:
            dict_wordcount[word] = {'Total Term Frequency': 1}
        else:
            dict_wordcount[word]['Total Term Frequency'] += 1

# In[]
""" --- document frequency and inverse document frequency --- """
# DF
for word in tqdm(dict_wordcount.keys()):
    for i in dict_text.keys():
        if word in dict_text[i]['text_preprocessed']:
            if 'Document Frequency' not in dict_wordcount[word].keys():
                dict_wordcount[word]['Document Frequency'] = 1
            else:
                dict_wordcount[word]['Document Frequency'] += 1

# IDF
for word in tqdm(dict_wordcount.keys()):
    DF = dict_wordcount[word]['Document Frequency']
    IDF = 1 + math.log(len(dict_text)/DF)
    dict_wordcount[word]['Inverse Document Frequency'] = IDF    
    

# In[]
""" --- remove the top 50 and DF<25 unigrams regarding DF --- """

# get dictionary of DF
dict_wordcount_DF = {}
for word in tqdm(dict_wordcount.keys()):
    dict_wordcount_DF[word] = dict_wordcount[word]['Document Frequency']
    
# get the list of unigrams regarding DF in descending order
list_wordcount_DF = sorted(dict_wordcount_DF.items(),key=lambda x:x[1],reverse=True)
# remove ''
for i in range(0,len(list_wordcount_DF)):
    if list_wordcount_DF[i][0] == '':
        break
list_wordcount_DF.remove(list_wordcount_DF[i])

# get index of 1st word with DF=25
length = 0 
for word in tqdm(list_wordcount_DF):
    if word[1] < 25:
        length = list_wordcount_DF.index(word)
        break

# top 100 tokens as features
list_wordcount_top100 = list_wordcount_DF[0:100]

# remove top 50 and DF<25 to obtain the controlled vocabulary
list_wordcount_DF = list_wordcount_DF[50:length]


# In[]
""" --- construct dictionary for controlled vocabulary --- """

dict_ctrlvoc_TF = {}
# count term frequency for each document
for i in tqdm(dict_text.keys()):
    dict_ctrlvoc_TF[i] = {}
    for word in list_wordcount_DF:
        dict_ctrlvoc_TF[i][word[0]] = dict_text[i]['text_preprocessed'].count(word[0])
        
# In[]
""" --- convert dictionary into dataframe ---"""
df_ctrlvoc_TF = pd.DataFrame.from_dict(dict_ctrlvoc_TF).T.astype(np.float64)

# In[]
""" --- calculate probabilities: TF/TTF --- """
# dataframe for saving probabilities of term frequencies
df_ctrlvoc_prob = df_ctrlvoc_TF.astype(np.float64)

# for each column (token)
for token in tqdm(df_ctrlvoc_prob.columns.values.tolist()):
    # obtain total term frequency for current column(token) from dictionary
    TTF = dict_wordcount[token]['Total Term Frequency']
    # calculate probability: TF(d)/TTF
    df_ctrlvoc_prob[token] = df_ctrlvoc_TF[token]/TTF
    
# In[]
""" --- calculate weight: TF*IDF --- """

# weight(term, document) = TF(term, document) * IDF(term)
df_ctrlvoc_weight = df_ctrlvoc_TF.astype(np.float64)

for token in tqdm(df_ctrlvoc_weight.columns.values.tolist()):
    # obtain inverse document frequency for current column(token) from dictionary
    IDF = dict_wordcount[token]['Inverse Document Frequency']
    # calculate weight(t,d) = TF(t,d) * IDF(t)
    df_ctrlvoc_weight[token] = df_ctrlvoc_TF[token] * IDF


# In[]
""" --- save two new variables into dict --- """
""" --- extract sentiment features as well --- """


for i in tqdm(df_ctrlvoc_TF.mean(axis=1).index):
    # save mean probability of TF/TTF
    dict_text[i]['mean_prob'] = df_ctrlvoc_TF.mean(axis=1)[i]
    # save weight = TF*IDF
    dict_text[i]['mean_weight'] = df_ctrlvoc_weight.mean(axis=1)[i]
    # sentiment analysis
    dict_text[i]['sentiment'] = TextBlob(dict_text[i]['text']).sentiment
    dict_text[i]['sentiment_polarity'] = dict_text[i]['sentiment'][0]
    dict_text[i]['sentiment_subjectivity'] = dict_text[i]['sentiment'][1]

# In[]
""" --- add tokens as features as well --- """
for word in tqdm(list_wordcount_top100):
    for i in dict_text.keys():
        if word[0] in dict_text[i]['text_preprocessed']:
            dict_text[i][word[0]] = 1
        else:
            dict_text[i][word[0]] = 0

# In[]
df_text = pd.DataFrame.from_dict(dict_text).T

# In[]
list_feature = ['textclean_length','text_length','mean_prob','mean_weight','sentiment_polarity','sentiment_subjectivity']

for word in list_wordcount_top100:
    list_feature.append(word[0])


# In[]
""" save features in df """
df_new = pd.concat([df_data,df_text[list_feature]],axis = 1)

# In[]
""" save df into csv """ 
df_new.to_csv(os.path.join(path,'Daily Twitter data','master_topics_with text features and tokens.csv'), index=False, quoting=1)










    
    
    