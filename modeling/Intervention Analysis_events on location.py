# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:10:21 2019

@author: Yichen Jiang
"""

"""
--- this file is for intervention analysis on how candidates' events affect fund raising in different location ---

"""
# In[]
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
from operator import itemgetter
import random

# In[]:
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
--- import data ---

"""
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science'
path_events = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Events'
path_finance = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Fundraising data'

filenames = os.listdir(path)
filenames_events = os.listdir(path_events)
filenames_finance = os.listdir(path_finance)

# In[]
"""
--- read events data (for sorting events location) ---

"""

for filename in filenames_events:
    if filename.endswith('.json'):
        file = open(os.path.join(path_events, filename), 'r', encoding = 'utf-8')
        for line in file.readlines():
            dict_events = json.loads(line)
            
# In[]:
"""
--- read dict_data file ---

"""

for filename in filenames_finance:
    if 'dict_data' in filename:
        file = open(os.path.join(path_finance, filename), 'r', encoding = 'utf-8')
        for line in file.readlines():
            dict_data = json.loads(line)

# In[]:
"""
--- read USA state initial data ---

"""

list_state = []
list_statename = []
count = 0

for filename in filenames:
    if 'State' in filename:
        file = open(os.path.join(path, filename), 'r',encoding='utf-8-sig')
        lines = csv.reader(file)
        for line in lines:
            if count == 0:
                count += 1
                continue
            else:
                list_state.append(line[1])
                list_statename.append(line)
                
# In[]:
"""
--- sort data by location ---

"""
# dictionary of data by states
dict_state = {}

for candidate in tqdm(dict_data['finance'].keys()):
    # add candidate in dict_state
    if candidate not in dict_state.keys():
        dict_state[candidate] = {}
        # add state in dict_state for current candidate
        for state in list_state:
            dict_state[candidate][state] = {}
        # for unrecognized state
        dict_state[candidate][' '] = {}
    for date in dict_data['finance'][candidate].keys():
        for attribute in dict_data['finance'][candidate][date].keys():
            # if current attribute is a state
            if attribute in dict_state[candidate].keys():
                if date not in dict_state[candidate][attribute].keys():
                    dict_state[candidate][attribute][date] = {}
                dict_state[candidate][attribute][date]['number_of_donors'] = \
                dict_data['finance'][candidate][date][attribute]
                dict_state[candidate][attribute][date]['amount_of_donations'] = \
                dict_data['finance'][candidate][date][str(attribute)+'_donation']
            elif attribute == 'time':
                for state in dict_state[candidate].keys():
                    dict_state[candidate][state][date][attribute] = \
                    dict_data['finance'][candidate][date][attribute]

# In[]:
"""
--- sort events data by location ---

"""
# a temporary dictionary which saves a list of potential locations of all events
dict_templocation = {}

for candidate in tqdm(dict_events.keys()):
    if str.upper(candidate) not in dict_templocation.keys():
        dict_templocation[str.upper(candidate)] = {}
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
                if temp_time not in dict_templocation[str.upper(candidate)].keys():
                    dict_templocation[str.upper(candidate)][temp_time] = []
                # check event location
                location = dict_events[candidate][year][month][day]['Location']
                for state in list_state:
                    if state in location:
                        dict_templocation[str.upper(candidate)][temp_time].append(state)
                # if none of them fit
                if dict_templocation[str.upper(candidate)][temp_time] == []:
                    dict_templocation[str.upper(candidate)][temp_time].append(location)
                    print(dict_events[candidate][year][month][day])
                    print('\n')

# In[]:
"""
--- manually revise missing location information ---

"""
dict_templocation['SANDERS']['20190407'] = ['IA']

dict_templocation['SANDERS']['20190405'] = ['IA']

dict_templocation['SANDERS']['20190310'] = ['NH']

dict_templocation['SANDERS']['20190302'] = ['NY']

dict_templocation['WARREN']['20190519'] = ['NH']

dict_templocation['WARREN']['20190318'] = ['MS']

dict_templocation['WARREN']['20190308'] = ['NY']

# In[]
"""
--- combine the information of event location with dict_state (donor state)

"""

for candidate in tqdm(dict_state.keys()):
    for state in dict_state[candidate].keys():
        if state == ' ':
            continue
        for date in dict_state[candidate][state].keys():
            for candidate_compare in dict_templocation.keys():
                # if candidate_compare had an event on current date
                # && if this event happend in current state
                if date in dict_templocation[candidate_compare].keys() and \
                state == dict_templocation[candidate_compare][date][0]:
                    dict_state[candidate][state][date][str(candidate_compare)+'_event'] = 1
                else:
                    dict_state[candidate][state][date][str(candidate_compare)+'_event'] = 0

# In[]
"""
--- location count ---

"""
dict_locationcount = {}

for candidate in dict_templocation.keys():
    if candidate not in dict_locationcount.keys():
        dict_locationcount[candidate] = {}
    for date in dict_templocation[candidate].keys():
        if dict_templocation[candidate][date][0] not in dict_locationcount[candidate].keys():
            dict_locationcount[candidate][dict_templocation[candidate][date][0]] = 1
        else:
            dict_locationcount[candidate][dict_templocation[candidate][date][0]] += 1
    
# In[]
"""
--- sorting data by descending order ---

"""
for candidate in dict_locationcount.keys():
    createVar['list_locationcount_'+str(candidate)] = sorted(dict_locationcount[candidate].items(),key=lambda x:x[1],reverse=True)

# In[]:
"""
--- sort values for plot ---

"""
list_candidate = ['BIDEN', 'SANDERS', 'WARREN', 'HARRIS']

# dictionary for saving plot values
dict_plot = {}

# number of states chosen
number_state = 3

for candidate in list_candidate:
    if candidate not in dict_plot.keys():
        dict_plot[candidate] = {}
    # find top 3 states with most number of events for each candidate
    for i in range(0, number_state):
        state = createVar['list_locationcount_'+str(candidate)][i][0]
        dict_plot[candidate][state] = {}
        # sort dictionary by key (date) in ascending order
        list_temp = sorted(dict_state[candidate][state].items(),key=lambda x:x[0],reverse=False)
        for line in list_temp:
            for variable in line[1].keys():
                if variable not in dict_plot[candidate][state].keys():
                    dict_plot[candidate][state][variable] = []
                dict_plot[candidate][state][variable].append(line[1][variable])

# In[]
# define check_state function for checking state whole name
def check_state(state,list_statename):
    state_name = ''
    for state_pair in list_statename:
        if state_pair[1] == state:
            state_name = state_pair[0]
    return state_name

# In[]
dict_size = {67: 18, 132: 16, 142: 16, 161: 14}

# In[]
"""
--- draw figures for 4 candidates regarding several important states ---
--- number of donors ---
--- amount of donations ---

"""

for candidate in dict_plot.keys():
    for state in dict_plot[candidate].keys():
        state_name = check_state(state,list_statename)
        # plot
        r_1 = randomcolor() 
        r_2 = randomcolor() 
        length = len(dict_plot[candidate][state]['time'])
        fig = plt.figure(dpi = 80, figsize = (0.5*length, 0.25*length))
        plt.title('Candidate: '+str(candidate)+', State: '+str(state_name), fontdict={'size':'40'})
        plt.subplot(1, 1, 1)
        
        values_1 = dict_plot[candidate][state]['amount_of_donations']
        values_2 = dict_plot[candidate][state]['number_of_donors']
        index = np.arange(length)
        plt.plot(index, values_1, label="Amount of donations per day", color=r_1)
        plt.plot(index, values_2, label="Number of donors per day", color=r_2)        
        #plot marker
        for i in range(length):
            marker = dict_plot[candidate][state][str(candidate)+'_event']
            plt.text(index[i], values_1[i], str(marker[i]), fontdict={'size':'20','color':'b'}) 
            
        plt.xlabel('Date', fontdict={'size':'40'})
        plt.ylabel('Amount of donations & Number of donors', fontdict={'size':'40'})
        font_size = dict_size[length]
        plt.xticks(index, dict_plot[candidate][state]['time'], fontsize=font_size, rotation = 45)
        
        plt.show()    

# In[]:
for candidate in dict_plot.keys():
    createVar['list_statecount_'+str(candidate)] = []
    for pair in createVar['list_locationcount_'+str(candidate)]:
        for state in list_statename:
            if state[1] == pair[0]:
                createVar['list_statecount_'+str(candidate)].append([state[0],pair[1]])
                break

# In[]
"""
--- save event data ---

"""

with open(os.path.join(path_events, 'Events.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_events, outfile, ensure_ascii=False)                 
                