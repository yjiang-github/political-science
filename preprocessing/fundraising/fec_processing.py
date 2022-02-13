# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:17:26 2020

@author: Yichen Jiang
"""

"""
--- this file is for processing the fec data ---

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
createVar = locals()

# In[]: # define plot color generator

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ''
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return '#'+color

# In[]
"""
--- define dict_candidates ---

"""
dict_candidates = {'biden':'joe','buttigieg':'pete','harris':'kamala','sanders':'bernie','warren':'elizabeth','yang':'andrew'}

# In[]:
"""
--- Define path ---
--- import FEC data ---

"""

path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science'

path_fec = os.path.join(path,'Fundraising data','2019-2020')


filenames_fec = os.listdir(path_fec)

dict_fecraw = {}

# In[]

list_attributes = ['contribution_receipt_date','transaction_id','contributor_name','contributor_zip','contribution_receipt_amount']

for filename in tqdm(filenames_fec):
    if filename.endswith('.csv') and '2020' not in filename and '2018' not in filename:
        candidate = filename.split('_')[0]
        print(candidate)
        print(filename,'\n')
        if candidate not in dict_fecraw.keys():
            dict_fecraw[candidate] = {}
            #dict_feccsv[candidate] = {}
                
        """
        # read csv into dataframe
        df_temp = pd.read_csv(os.path.join(path_fec, filename),low_memory=False)
        if len(dict_feccsv[candidate]) == 0:
            dict_feccsv[candidate] = df_temp
        else:
            dict_feccsv[candidate] = pd.concat([dict_feccsv[candidate],df_temp],ignore_index=True)
        """
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
                index_contributor_name = list_columnname.index('contributor_name')
                index_contributor_zip = list_columnname.index('contributor_zip')
                index_contribution_receipt_amount = list_columnname.index('contribution_receipt_amount')
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
                for attribute in list_attributes:
                    dict_fecraw[candidate][date][transaction_id][attribute] = line[createVar.get('index_'+str(attribute))]
        
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
            if dict_donor[candidate][name_zip][transaction_id]['contribution_receipt_amount'] != '':
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
                if dict_donor[candidate][name_zip][transaction_id]['contribution_receipt_amount'] != '':
                    dict_fec[candidate][date]['newdonation_amount'] += float(dict_donor[candidate][name_zip][transaction_id]['contribution_receipt_amount'])
         # clear dict_temp
        dict_temp = {}

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
              'yang':'2019-01-01',
              'booker':'2019-02-01',
              'gabbard':'2019-01-11',
              'klobuchar':'2019-02-10',
              "o'rourke":'2019-03-14',
              'steyer':'2019-07-09',
              'bloomberg':'2019-11-24'} # actually yang started much more eailer than others, Feb 2018

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

"""
--- sort all candidates' data by category ---

"""
dict_fec_type = {}

for column in dict_fec_df['biden'].columns:
    if column not in dict_fec_type.keys():
        dict_fec_type[column] = {}
    for candidate in dict_fecraw.keys():
        df_temp = dict_fec_df[candidate][column]
        if len(dict_fec_type[column]) == 0:
            dict_fec_type[column] = df_temp
        else:
            dict_fec_type[column] = pd.concat([dict_fec_type[column],df_temp],axis=1,sort=True)
    
    dict_fec_type[column].columns = dict_fecraw.keys()

# In[]
"""
--- remove the influence of startday ---

"""
for column in dict_fec_type.keys():
     for candidate in dict_fec_type[column].columns:
         start_date = dict_start[candidate]
         dict_fec_type[column].loc[start_date,candidate] = 0

# In[]
"""
--- replace nan with 0 ---

"""
for column in dict_fec_type.keys():
    dict_fec_type[column]=dict_fec_type[column].where(dict_fec_type[column].notnull(), 0)


# In[]
"""
--- normalize data into the scale of 0-1 by each day ---

"""
for column in tqdm(dict_fec_type.keys()):
    for index in dict_fec_type[column].index:
        Max = dict_fec_type[column].loc[index].max()
        Min = dict_fec_type[column].loc[index].min()
        for candidate in dict_fec_type[column].columns:
            dict_fec_type[column].loc[index][candidate] = \
            (dict_fec_type[column].loc[index][candidate]-Min)/(Max-Min)

# In[]
dict_lim = {'donor_num':25000,
            'newdonor_num':3500,
            'donation_amount':2000000,
            'newdonation_amount':800000}

dict_color = {'biden':'limegreen',
              'buttigieg':'cyan',
              'harris':'orange',
              'sanders':'red',
              'warren':'blue',
              'yang':'magenta'}

    
# In[]
"""
--- plot ---
--- by categories ---

"""




# find the index of 2019-01-01
index_start = list(dict_fec_type['donor_num'].index).index('2019-01-01')


for column in dict_fec_type.keys():
    
    plt.style.use('ggplot')

    fig = plt.figure(dpi = 80, figsize = (75, 25))
    plt.title(column,  fontsize=50)
     
    index = np.arange(len(dict_fec_type['donor_num'].iloc[index_start:]))
    list_legend = []
    
    # plot each candidate
    for candidate in dict_fec_type[column].columns:
        #r = randomcolor()
        createVar[str(candidate).upper()], = plt.plot(index, dict_fec_type[column][candidate].iloc[index_start:],label=candidate+':'+column,color=dict_color[candidate])
        list_legend.append(createVar[str(candidate).upper()])
        
    # create list for xticks, list for peak numbers, and setup initial month
    list_x = ['' for i in range(len(dict_fec_type['donor_num'].iloc[index_start:]))]
    month = '00'
    for num in range(len(dict_fec_type['donor_num'].iloc[index_start:,:])):
        # overwrite if a new month has been detected
        if dict_fec_type[column][candidate].iloc[[index_start+num]].index.values[0][5:7] != month:
            # extract month
            month = dict_fec_type[column][candidate].iloc[[index_start+num]].index.values[0][5:7]
            list_x[num] = dict_fec_type[column][candidate].iloc[[index_start+num]].index.values[0][5:7]
        # overwrite list_x[num] if it is on debate dates, and, add vertical line for debate dates
        if dict_fec_type[column][candidate].iloc[[index_start+num]].index.values[0] in list_dates:
            list_x[num] = dict_fec_type[column][candidate].iloc[[index_start+num]].index.values[0][5:10]
            debate_dates = plt.axvline(num, color='k', linestyle='--', label = 'Debate Dates')
        # overwrite if it is on a candidate's start date
        for candidate in dict_candidates.keys():
            if dict_fec_type[column][candidate].iloc[[index_start+num]].index.values[0] == dict_start[candidate]:
                # add vertical line
                #r = randomcolor()
                createVar[str(candidate).upper()+'_start date'] = plt.axvline(num, color=dict_color[candidate], linestyle='--', label = candidate+':start date')
                list_legend.append(createVar[str(candidate).upper()+'_start date'])
    list_legend.append(debate_dates)
    
    plt.xticks(index, list_x,fontsize = 16,rotation=45)
    plt.yticks(fontsize = 50)
    plt.ylim(0,dict_lim[column])
    plt.legend(handles = list_legend,loc='upper right',fontsize = 35)
    plt.savefig(os.path.join(path_fec,'plots', column+'.png'))
    plt.show()

# In[]
dict_count = {}
for column in dict_fec_type.keys():
    if column not in dict_count.keys():
        dict_count[column] = {}
        for candidate in dict_candidates.keys():
            dict_count[column][candidate] = {'count':0,'date':[]}
    for index in dict_fec_type[column].index:
        if index < '2019-01-01':
            continue
        # find the biggest number on this day
        candidate = dict_fec_type[column].loc[index][dict_fec_type[column].loc[index] == dict_fec_type[column].loc[index].max()].index[0]
        dict_count[column][candidate]['count'] += 1
        dict_count[column][candidate]['date'].append(index)

# In[]
"""
--- export dfs in dict_fec_type ---

"""

for column in dict_fec_type.keys():
    dict_fec_type[column].to_csv(os.path.join(path_fec,'amount and number', column +'.csv'), index=True, quoting=1)

# In[]
"""
--- import .csv into dict_fec_type ---

"""
filenames_processed = os.listdir(os.path.join(path_fec,'processed_data'))
dict_fec_type = {}
for filename in filenames_processed:
    if filename.endswith('csv'):
        dict_fec_type[filename.strip('.csv')] = pd.read_csv(os.path.join(path_fec,'processed_data',filename),index_col = 0)

# In[]
"""
--- data aggregation ---

"""
dict_databyweek = {}
# start on 20190101


for column in dict_fec_type.keys():
    if column not in dict_databyweek.keys():
        dict_databyweek[column] = {}
    for candidate in dict_candidates.keys():
        count = 0
        if candidate not in dict_databyweek[column].keys():
            dict_databyweek[column][candidate] = {}
            for index in dict_fec_type[column].index.values:
                if '2019-01-01' <= index <= '2019-12-31':
                    #print(index)
                    # start of a week
                    if count == 0:
                        start_date = index
                        dict_databyweek[column][candidate][start_date] = dict_fec_type[column][candidate].loc[index]
                        count += 1
                    # end of a week
                    elif count == 6:
                        dict_databyweek[column][candidate][start_date] += dict_fec_type[column][candidate].loc[index]
                        count = 0
                    # middle of a week
                    else:
                        dict_databyweek[column][candidate][start_date] += dict_fec_type[column][candidate].loc[index]
                        count += 1

# In[]
dict_databyweek_df = {}
for column in dict_databyweek.keys():
    dict_databyweek_df[column] = pd.DataFrame.from_dict(dict_databyweek[column])
    
# In[]
"""
--- plot ---

"""

for column in dict_databyweek_df.keys():
    list_legend = []
    
    plt.style.use('ggplot')

    fig = plt.figure(dpi = 80, figsize = (75, 25))
    plt.title(column,  fontsize=50)
     
    index = np.arange(len(dict_databyweek_df[column]))
        # plot each candidate
    for candidate in dict_databyweek_df[column].columns:
        createVar[str(candidate).upper()], = plt.plot(index, dict_databyweek_df[column][candidate],label=candidate+':'+column,color=dict_color[candidate])
        list_legend.append(createVar[str(candidate).upper()])
        
    plt.xticks(index,dict_databyweek_df[column].index,fontsize = 20,rotation = 45)
    plt.yticks(fontsize = 50)
    plt.legend(handles = list_legend,loc='upper right',fontsize = 35)
    plt.savefig(os.path.join(path_fec,'plots', column+'.png'))
    plt.show()

    
