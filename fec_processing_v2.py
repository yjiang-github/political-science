# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:55:03 2020

@author: Yichen Jiang
"""

"""
--- this file is for processing candidates' donation data for percentage donation 
    per candidate per day ---
    
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

# In[]
""" --- import data --- """
path_fec = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Fundraising data\2019-2020'

filenames_fec = os.listdir(path_fec)

list_candidates = ['biden','sanders','warren','buttigieg','harris','yang','bloomberg',
                   'klobuchar',"o'rourke",'gabbard','booker','steyer']

# In[]

dict_amount = {}
list_processed = []


# In[]
""" --- data processing --- """
for filename in tqdm(filenames_fec):
    if filename.endswith('.csv'):
        if '2020' in filename:
            continue
        if filename in list_processed:
            continue
        
        curr_candidate = filename.split('_')[0]
        
        # read csv into dataframe
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
                index_contribution_receipt_amount = list_columnname.index('contribution_receipt_amount')
                # not the 1st line
                count += 1       
            else:
                if line[index_contribution_receipt_amount] != '':
                    contribution_receipt_amount = float(line[index_contribution_receipt_amount])
                contribution_receipt_date = line[index_contribution_receipt_date]
                year = contribution_receipt_date[0:4]
                month = contribution_receipt_date[5:7]
                day = contribution_receipt_date[8:10]      
                date = year+'-'+month+'-'+day        
                if date not in dict_amount.keys():
                    dict_amount[date] = {}
                    for candidate in list_candidates:
                        dict_amount[date][candidate] = 0.0
                    dict_amount[date]['sum'] = 0.0
                dict_amount[date][curr_candidate] += contribution_receipt_amount
                dict_amount[date]['sum'] += contribution_receipt_amount
                
        list_processed.append(filename)
        
# In[]
""" --- calculate percentage --- """
dict_percentage = {}

for date in tqdm(dict_amount.keys()):
    dict_percentage[date] = {}
    for candidate in dict_amount[date].keys():
        if candidate != 'sum':
            dict_percentage[date][candidate] = dict_amount[date][candidate]/dict_amount[date]['sum']

# In[]
            
df_amount = pd.DataFrame.from_dict(dict_amount).T
df_amount.sort_index(inplace=True)

df_percentage = pd.DataFrame.from_dict(dict_percentage).T
df_percentage.sort_index(inplace=True)

# In[]

df_amount.to_csv(os.path.join(path_fec,'amount and percentage', 'donation amount per candidate per day.csv'), index=True, quoting=1)
df_percentage.to_csv(os.path.join(path_fec,'amount and percentage', 'percentage per candidate per day.csv'), index=True, quoting=1)
