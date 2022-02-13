# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:42:04 2019

@author: Yichen Jiang
"""

import os
from tqdm import tqdm
import json
import datetime
import string
import numpy as np
import pandas as pd
import pyflux as pf
import matplotlib.pyplot as plt
import csv
import random

# In[]:
createVar = locals()

# In[]
def takeFirst(elem):
    return elem[0]

# In[]
"""
--- define str-to-date function ---

"""
def str2date(str):
    date = datetime.datetime.strptime(str,"%Y%m%d")
    return date

# In[]: # define plot color generator
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ''
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return '#'+color

# In[]
"""
--- import data ---

"""
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science'

path_finance = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Fundraising data'
path_events = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Events'

filenames_finance = os.listdir(path_finance)
filenames_events = os.listdir(path_events)

# In[]
"""
--- read events data ---

"""
for filename in filenames_events:
    if filename.endswith('.json'):
        file = open(os.path.join(path_events, filename), 'r', encoding = 'utf-8')
        for line in file.readlines():
            dict_events = json.loads(line)
            
# In[]
"""
--- read finance data ---

"""
for filename in filenames_finance:
    if filename.endswith('.json'):
        file = open(os.path.join(path_finance, filename),'r',encoding='utf-8')
        for line in file.readlines():
            if 'transaction' in filename:
                dict_finance = json.loads(line)
            elif 'donations' in filename:
                dict_donations = json.loads(line)

# In[]
"""
--- sorting finance data by each donors with their transactions ---
--- by contributor_name

"""
# create dictionary for donors information
dict_donor_name = {}

# sorting data by each donor
for candidate in tqdm(dict_finance.keys()):
    for year in dict_finance[candidate].keys():
        for month in dict_finance[candidate][year].keys():
            for day in dict_finance[candidate][year][month].keys():
                for transaction_id in dict_finance[candidate][year][month][day].keys():
                    # get donor's name
                    contributor_name = dict_finance[candidate][year][month][day][transaction_id]['contributor_name']
                    contributor_zip = dict_finance[candidate][year][month][day][transaction_id]['contributor_zip']
                    name_zip = str(contributor_name)+'_'+str(contributor_zip)
                    transaction_time = dict_finance[candidate][year][month][day][transaction_id]['contribution_receipt_date'][0:10]
                    # save transaction information
                    if contributor_name not in dict_donor_name.keys():
                        dict_donor_name[contributor_name] = {}
                    # save name_zip for this contributor_name
                    if name_zip not in dict_donor_name[contributor_name].keys():
                        dict_donor_name[contributor_name][name_zip] = {}
                    # save this transaction
                    dict_donor_name[contributor_name][name_zip][str(transaction_time)+'_'+str(transaction_id)] = \
                    dict_finance[candidate][year][month][day][transaction_id]

# In[]:
"""
--- check dict_donor_name to see if we have donors that donate multiple times---

"""
# create dictionary for donors with multiple donations
dict_donor_name_multiple = {}

for contributor_name in dict_donor_name.keys():
    for name_zip in dict_donor_name[contributor_name].keys():
        if len(dict_donor_name[contributor_name][name_zip]) >= 2:
            dict_donor_name_multiple[name_zip] = dict_donor_name[contributor_name][name_zip]

# In[]:
"""
--- check if any donor donate to 2 or more candidates ---

"""
dict_donor_name_mulcom = {}

list_committee_id = []

for donor_name in dict_donor_name_multiple.keys():
    for transaction in dict_donor_name_multiple[donor_name].keys():
        if dict_donor_name_multiple[donor_name][transaction]['committee_id'] not in list_committee_id:
            list_committee_id.append(dict_donor_name_multiple[donor_name][transaction]['committee_id'])
    if len(list_committee_id) >= 2:
        dict_donor_name_mulcom[donor_name] = dict_donor_name_multiple[donor_name]
    list_committee_id = []

# In[]:
"""
--- sorting data for intervention analysis ---
--- finance data ---
--- find new donors, new donations, small donors and big donors ---

"""
# import USA State initials
list_state = []
count = 0
file = open(os.path.join(path, 'USA State Abbreviations.csv'), 'r',encoding='utf-8-sig')
lines = csv.reader(file)
for line in lines:
    if count == 0:
        count += 1
        continue
    else:
        list_state.append(line[1])

# In[]
# dictionary of start date of each candidate
dict_starttime = {'BIDEN': str2date('20190425'), 'HARRIS': str2date('20190121'), 'SANDERS': str2date('20190219'), \
             'WARREN': str2date('20190209')}

#dictionary for committee_id of each candidate
dict_committee_id = {'BIDEN': 'C00703975', 'HARRIS': 'C00694455', 'SANDERS': 'C00696948', 'WARREN': 'C00693234'}

# create dictionary for saving filtered finance data
dict_data = {'finance': {}, 'events': {}}

# temp time str
temp_time = ''

# number of small donors
small_donor = 0
# number of big donors
big_donor = 0
# number of new donors
new_donor = 0

# amount of donations received from new donors
new_amount = float(0)

for candidate in tqdm(dict_finance.keys()):
    if candidate not in dict_data['finance'].keys():
        dict_data['finance'][candidate] = {}
    for year in dict_finance[candidate].keys():
        if year != '2019':
            continue
        for month in dict_finance[candidate][year].keys():
            for day in dict_finance[candidate][year][month].keys():
                # save the time of current date into str
                temp_time = str(year)
                if int(month) < 10:
                    temp_time += '0'+str(month)
                else: temp_time += str(month)
                if int(day) < 10:
                    temp_time += '0'+str(day)
                else: temp_time += str(day)
                # check if current date came after the start date of current candidate
                if str2date(temp_time) < dict_starttime[candidate]:
                    continue
                
                # save #_of_donors & donation_amount
                if temp_time not in dict_data['finance'][candidate].keys():
                    dict_data['finance'][candidate][temp_time] = {'number_of_transactions': len(dict_finance[candidate][year][month][day])}
                dict_data['finance'][candidate][temp_time]['donation_amount'] = dict_donations[candidate][year][month][day]
                # save USA state initials in dict_data
                for state in list_state:
                    # number of donors and donation amount
                    dict_data['finance'][candidate][temp_time][state] = 0
                    dict_data['finance'][candidate][temp_time][str(state)+'_donation'] = float(0)
                dict_data['finance'][candidate][temp_time][' '] = 0
                dict_data['finance'][candidate][temp_time][' '+'_donation'] = float(0)
                
                # check donor status
                for transaction_id in dict_finance[candidate][year][month][day].keys():
                    # save number of donors and amount of donations by states
                    contributor_state = dict_finance[candidate][year][month][day][transaction_id]['contributor_state']
                    donation_amount = float(dict_finance[candidate][year][month][day][transaction_id]['contribution_receipt_amount'])
                    if contributor_state not in dict_data['finance'][candidate][temp_time].keys():
                        dict_data['finance'][candidate][temp_time][' '] += 1
                        dict_data['finance'][candidate][temp_time][' '+'_donation'] = donation_amount
                    else:
                        dict_data['finance'][candidate][temp_time][contributor_state] += 1
                        dict_data['finance'][candidate][temp_time][str(contributor_state)+'_donation'] += donation_amount
                    # check donation amount on current transaction
                    if donation_amount < float(200):
                        small_donor += 1
                    else: big_donor += 1
                    # check if current donor is new to the current candidate
                    # get name_zip
                    contributor_name = dict_finance[candidate][year][month][day][transaction_id]['contributor_name']
                    contributor_zip = dict_finance[candidate][year][month][day][transaction_id]['contributor_zip']
                    name_zip = str(contributor_name)+'_'+str(contributor_zip)
                    # check if current donor has donated twice or more
                    # if not
                    if name_zip not in dict_donor_name_multiple.keys():
                        new_donor += 1
                        new_amount += donation_amount
                    else: # get the time & committee if of current transaction
                        current_time = datetime.datetime.strptime(dict_finance[candidate][year][month][day][transaction_id]['contribution_receipt_date'], "%Y-%m-%d %H:%M:%S")
                        current_committee_id = dict_finance[candidate][year][month][day][transaction_id]['committee_id']
                        # count how many transactions have been compared
                        count = 0
                        for transaction in dict_donor_name_multiple[name_zip].keys():
                            count += 1
                            # check if donations were received by the same candidate
                            if current_committee_id == dict_donor_name_multiple[name_zip][transaction]['committee_id']:
                                # get time and compare with the time of current transaction
                                transaction_time = datetime.datetime.strptime(dict_donor_name_multiple[name_zip][transaction]['contribution_receipt_date'], "%Y-%m-%d %H:%M:%S")
                                # if has donated to current candidate before
                                if transaction_time < current_time: # old donor
                                    break
                                else: continue
                        # check if all transactions were compared, or the for-loop has been breaked (because of old donor)
                        # no previous transaction has been found -> new donor
                        if count == len(dict_donor_name_multiple[name_zip]):
                            new_donor += 1
                            new_amount += donation_amount
                
                # save donor and donation data
                dict_data['finance'][candidate][temp_time]['number_of_small_donors'] = small_donor
                dict_data['finance'][candidate][temp_time]['number_of_big_donors'] = big_donor
                dict_data['finance'][candidate][temp_time]['number_of_new_donors'] = new_donor
                dict_data['finance'][candidate][temp_time]['amount_of_new_donations'] = round(new_amount, 2)
                
                # round donation by states
                for attribute in dict_data['finance'][candidate][temp_time].keys():
                    if '_donation' in attribute:
                        dict_data['finance'][candidate][temp_time][attribute] = \
                        round(dict_data['finance'][candidate][temp_time][attribute], 2)
                        
                # clear variables
                small_donor = 0
                big_donor = 0
                new_donor = 0
                new_amount = float(0)
                
# In[]
"""
--- sorting data for intervention analysis ---
--- events data ---

"""

# temp time str
temp_time = ''

# number of guests
number_of_guests = ''


for candidate in tqdm(dict_events.keys()):
    if str.upper(candidate) not in dict_data['events'].keys():
        dict_data['events'][str.upper(candidate)] = {}
    for year in dict_events[candidate].keys():
        for month in dict_events[candidate][year].keys():
            # since current data stops on 20190630:
            if int(month) >= 7:
                continue
            for day in dict_events[candidate][year][month].keys():
                # save the time of current date into str
                temp_time = str(year)
                if int(month) < 10:
                    temp_time += '0'+str(month)
                else: temp_time += str(month)
                if int(day) < 10:
                    temp_time += '0'+str(day)
                else: temp_time += str(day)
                
                # clear variable
                number_of_guests = ''
                # get number of guests
                temp_number = dict_events[candidate][year][month][day]['Number of Guests'].split(' ')[0]
                if temp_number == 'Hosted':
                    number_of_guests = 0
                else:
                    for elem in temp_number:
                        if elem not in string.punctuation:
                            number_of_guests += elem

                # save time and number of guests
                if temp_time not in dict_data['events'][str.upper(candidate)].keys():
                    dict_data['events'][str.upper(candidate)][temp_time] = {'number_of_guests': int(number_of_guests)}

# add number of audiences for democratic debate
number_of_guests = int(8670000/2 + 5870000 + 719000)
for candidate in dict_data['events'].keys():
    current_number = 0
    if '20190627' in dict_data['events'][candidate].keys():
        current_number = dict_data['events'][candidate]['20190626']['number_of_guests']
    dict_data['events'][candidate]['20190626'] = {'number_of_guests': number_of_guests + current_number}

# In[]
# add time 
for candidate in dict_data['finance'].keys():
    for date in dict_data['finance'][candidate].keys():
        dict_data['finance'][candidate][date]['time'] = int(date)
        
# In[]
"""            
--- sorting data for intervention analysis ---
--- combine finance data with events data(intervention) ---

"""
for candidate in dict_data['finance'].keys():
    for date in dict_data['finance'][candidate].keys():
        # check if this date has any candidate's event
        for check_candidate in dict_data['events'].keys():
            if date in dict_data['events'][check_candidate].keys():
                dict_data['finance'][candidate][date][str(check_candidate)+'_event'] = 1
                dict_data['finance'][candidate][date][str(check_candidate)+'_event_guests'] = \
                dict_data['events'][check_candidate][date]['number_of_guests']
            # if any candidate start campaign on current date
            elif str2date(date) == dict_starttime[check_candidate]:
                dict_data['finance'][candidate][date][str(check_candidate)+'_event'] = 1
                dict_data['finance'][candidate][date][str(check_candidate)+'_event_guests'] = 0
            else:
                dict_data['finance'][candidate][date][str(check_candidate)+'_event'] = 0
                dict_data['finance'][candidate][date][str(check_candidate)+'_event_guests'] = 0

# In[]:
"""
--- save dict_data ---

"""
with open(os.path.join(path_finance, 'dict_data.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_data, outfile, ensure_ascii=False)
    
# In[]
"""
--- fit with ARIMAX model with intervention ---

"""
dict_ARIMAX = {}
# convert all the str-date into num-date
for candidate in dict_data['finance'].keys():
    if candidate not in dict_ARIMAX.keys():
        dict_ARIMAX[candidate] = {}
    for date in dict_data['finance'][candidate].keys():
        dict_ARIMAX[candidate][int(date)] = dict_data['finance'][candidate][date]

# save dict_ARIMAX

# In[]
# create dataframe
for candidate in dict_starttime.keys():
    createVar['df_'+str(candidate)] = pd.DataFrame.from_dict(dict_ARIMAX[str(candidate)]).T
    createVar['df_'+str(candidate)] = createVar['df_'+str(candidate)].sort_index(axis=0, ascending=True, by=['time'])

#df_BIDEN = pd.DataFrame.from_dict(dict_ARIMAX['BIDEN']).T
    
# In[]
# standardize the data
for candidate in dict_data['finance'].keys():
    for date in dict_data['finance'][candidate].keys():
        for variable in dict_data['finance'][candidate][date].keys():
            if 'event' in variable and 'guests' not in variable:
                continue
            createVar['df_'+str(candidate)][str(variable)] = \
            (createVar['df_'+str(candidate)][str(variable)]-createVar['df_'+str(candidate)][str(variable)].min())/\
            (createVar['df_'+str(candidate)][str(variable)].max()-createVar['df_'+str(candidate)][str(variable)].min())
        break
    
# In[]
# create model
model = 'donation_amount~1'
for date in dict_ARIMAX['HARRIS'].keys():
    for variable in dict_ARIMAX['HARRIS'][date].keys():
        if variable == 'donation_amount':
            continue
        else: model += '+'+str(variable)
    break

# In[]
ARIMAX_model = pf.ARIMAX(data=df_WARREN, formula=model,ar=1, integ=1, ma=3,family=pf.Normal())
ARIMAX_result = ARIMAX_model.fit("MLE")
ARIMAX_result.summary()

# In[]
"""
--- plot the data ---

"""

mu, Y = ARIMAX_model._model(beta=ARIMAX_model.latent_variables.get_z_values())
date_index = ARIMAX_model.index[max(ARIMAX_model.ar, ARIMAX_model.ma):ARIMAX_model.data.shape[0]]
length = len(date_index)
x_index = np.linspace(0, length-1, length)
lag = max(ARIMAX_model.ar, ARIMAX_model.ma)
temp_list = sorted([i for i in dict_ARIMAX['HARRIS'].keys()], reverse=False)
list_x = [temp_list[i+lag] for i in range(len(temp_list)-lag)]
values_to_plot = ARIMAX_model.link(mu)

# In[]
plt.figure(dpi = 100, figsize = (15,10))
plt.plot(x_index, Y, label='Data',linestyle = "-")
plt.plot(x_index, list(values_to_plot), label='ARIMA model', c='black',linestyle = "-")
plt.title(ARIMAX_model.data_name)
plt.legend(loc=2)

#plt.xticks(list_x)   
plt.show()

# In[]
"""
--- predict ---

"""
#ARIMAX_model.plot_predict(h=10, oos_data=df_HARRIS.iloc[-12:], past_values=100, figsize=(15,5))


# In[]
"""
--- plot dictionary ---

"""

dict_plot = {}

for candidate in dict_donations.keys():
    if candidate not in dict_plot.keys():
        dict_plot[candidate] = {}
        list_temp = sorted(dict_data['finance'][candidate].items(),key=lambda x:x[0],reverse=False)
        for line in list_temp:
            for variable in line[1].keys():
                if variable not in dict_plot[candidate].keys():
                    dict_plot[candidate][variable] = []
                dict_plot[candidate][variable].append(line[1][variable])
        
# In[]
"""
--- plot values ---
--- new donors ---
--- amount of new donations ---
"""
# dictionary of font size
dict_size = {67: 18, 132: 16, 142: 16, 161: 14}

for candidate in dict_plot.keys():
    r = randomcolor() 
    for variable in dict_plot[candidate].keys():
        if variable == 'number_of_new_donors' or variable == 'amount_of_new_donations':
            # plot
            length = len(dict_plot[candidate]['time'])
            font_size = dict_size[length]
            fig = plt.figure(dpi = 80, figsize = (0.5*length, 0.25*length))
            plt.title('Candidate: '+str(candidate)+ ','+str(variable), fontdict={'size':'40'})
            plt.subplot(1, 1, 1)
            
            values = dict_plot[candidate][variable]
            index = np.arange(length)
            plt.plot(index, values, label=str(variable), color=r)       
            #plot marker
            for i in range(length):
                marker = dict_plot[candidate][str(candidate)+'_event']
                plt.text(index[i], values[i], str(marker[i]), fontdict={'size':'20','color':'b'}) 
                
            plt.xlabel('Date', fontdict={'size':'40'})
            plt.ylabel(str(variable), fontdict={'size':'40'})
            plt.xticks(index, dict_plot[candidate]['time'], fontsize=font_size, rotation = 45)
            plt.yticks(fontsize = 25)
            plt.show()    

# In[]
"""
--- ARIMA model for new donations ---

"""
# In[]
"""
--- fit with ARIMAX model with intervention ---

"""
dict_ARIMAX = {}
# convert all the str-date into num-date
for candidate in dict_data['finance'].keys():
    if candidate not in dict_ARIMAX.keys():
        dict_ARIMAX[candidate] = {}
    for date in dict_data['finance'][candidate].keys():
        dict_ARIMAX[candidate][int(date)] = dict_data['finance'][candidate][date]

# save dict_ARIMAX

# In[]
# create dataframe
for candidate in dict_starttime.keys():
    createVar['df_'+str(candidate)] = pd.DataFrame.from_dict(dict_ARIMAX[str(candidate)]).T
    createVar['df_'+str(candidate)] = createVar['df_'+str(candidate)].sort_index(axis=0, ascending=True, by=['time'])

#df_BIDEN = pd.DataFrame.from_dict(dict_ARIMAX['BIDEN']).T
    
# In[]
# standardize the data
for candidate in dict_data['finance'].keys():
    for date in dict_data['finance'][candidate].keys():
        for variable in dict_data['finance'][candidate][date].keys():
            if 'event' in variable and 'guests' not in variable:
                continue
            createVar['df_'+str(candidate)][str(variable)] = \
            (createVar['df_'+str(candidate)][str(variable)]-createVar['df_'+str(candidate)][str(variable)].min())/\
            (createVar['df_'+str(candidate)][str(variable)].max()-createVar['df_'+str(candidate)][str(variable)].min())
        break
    
# In[]
# create model
model = 'amount_of_new_donations~1'
for date in dict_ARIMAX['HARRIS'].keys():
    for variable in dict_ARIMAX['HARRIS'][date].keys():
        if variable == 'amount_of_new_donations':
            continue
        else: model += '+'+str(variable)
    break
# In[]
# create model v2: candidate event
model = 'amount_of_new_donations~1'
for candidate in dict_ARIMAX.keys():
    model += '+'+str(candidate)+'_event'
    
# In[]
# fit the model
ARIMAX_model = pf.ARIMAX(data=df_SANDERS, formula=model,ar=2, integ=0, ma=2,family=pf.Normal())
ARIMAX_result = ARIMAX_model.fit("MLE")
ARIMAX_result.summary()

# In[]
"""
--- plot the data ---

"""

mu, Y = ARIMAX_model._model(beta=ARIMAX_model.latent_variables.get_z_values())
date_index = ARIMAX_model.index[max(ARIMAX_model.ar, ARIMAX_model.ma):ARIMAX_model.data.shape[0]]
length = len(date_index)
x_index = np.linspace(0, length-1, length)
lag = max(ARIMAX_model.ar, ARIMAX_model.ma)
temp_list = sorted([i for i in dict_ARIMAX['HARRIS'].keys()], reverse=False)
list_x = [temp_list[i+lag] for i in range(len(temp_list)-lag)]
values_to_plot = ARIMAX_model.link(mu)

# In[]
plt.figure(dpi = 100, figsize = (15,10))
plt.plot(x_index, Y, label='Data',linestyle = "-")
plt.plot(x_index, list(values_to_plot), label='ARIMA model', c='black',linestyle = "-")
plt.title(ARIMAX_model.data_name)
plt.legend(loc=2)

#plt.xticks(list_x)   
plt.show()










# In[]
"""
--- basic statistics of donors ---

"""
# number of unique donors identified
donor_count = 0

for donor_name in dict_donor_name.keys():
    donor_count += len(dict_donor_name[donor_name])

print(donor_count)




# In[]
"""
--- backup code ---

"""

"""
--- sorting finance data by each donors with their transactions ---
--- by contributor_id ---
--- failed since contributor_id are missing for most of the individual contributors ---

"""
# create dictionary for donors information
dict_donor_id = {}

# sorting data by each donor
for candidate in dict_finance.keys():
    for year in dict_finance[candidate].keys():
        for month in dict_finance[candidate][year].keys():
            for day in dict_finance[candidate][year][month].keys():
                for transaction_id in dict_finance[candidate][year][month][day].keys():
                    # get donor's id
                    contributor_id = dict_finance[candidate][year][month][day][transaction_id]['contributor_id']
                    if contributor_id not in dict_donor_id.keys():
                        dict_donor_id[contributor_id] = {}
                    if year not in dict_donor_id[contributor_id].keys():
                        dict_donor_id[contributor_id][year] = {}
                    if month not in dict_donor_id[contributor_id][year].keys():
                        dict_donor_id[contributor_id][year][month] = {}
                    if day not in dict_donor_id[contributor_id][year][month].keys():
                        dict_donor_id[contributor_id][year][month][day] = {}
                    # save transaction information
                    transaction_name = str(candidate)+'_'+str(transaction_id)
                    dict_donor_id[contributor_id][year][month][day][transaction_name] = \
                    dict_finance[candidate][year][month][day][transaction_id]

