# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:51:04 2019

@author: Yichen Jiang
"""
"""
--- this file is for importing candidates' events from .csv file, sort them by each candidate by each date
and save them in .json file ---

"""
# In[]
import csv
import json
import os

# In[]
createVar = locals()

# In[]:
"""
--- import data ---

"""

path_events = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Events'

filenames = os.listdir(path_events)

# In[]:
# list of candidates
list_candidates = ['Biden', 'Sanders', 'Warren', 'Harris']

# In[]
"""
--- read data ---

"""
# list of column names
list_columnnames = []

# dictionary of events
dict_events = {}

for filename in filenames:
    if filename.endswith('.csv'):
        for candidate in list_candidates:
            # check which candidate the event schedule belongs to
            if candidate in filename:
                # create key for this candidate
                dict_events[str(candidate)] = {}
                break
        
        count = 0
        
        file = open(os.path.join(path_events, filename), 'r',encoding='utf-8-sig')
        lines = csv.reader(file)
        for line in lines:
            # 1st row with column names
            if count == 0:
                list_columnnames = line
                index_date = list_columnnames.index('Date')
                count += 1
            else:
                date = line[index_date]
                year = int(date.split('/')[2])
                month = int(date.split('/')[0])
                day = int(date.split('/')[1])
                if year not in dict_events[str(candidate)].keys():
                    dict_events[str(candidate)][year] = {}
                if month not in dict_events[str(candidate)][year].keys():
                    dict_events[str(candidate)][year][month] = {}
                if day not in dict_events[str(candidate)][year][month].keys():
                    dict_events[str(candidate)][year][month][day] = {}
                for i in range(len(line)):
                    columnname = list_columnnames[i]
                    if columnname == 'Time&Guests':
                        dict_events[str(candidate)][year][month][day]['Time'] = \
                        line[i].split('·')[0][0:len(line[i].split('·')[0])-1] # remove the space at the end
                        dict_events[str(candidate)][year][month][day]['Number of Guests'] = \
                        line[i].split('·')[1][1:len(line[i].split('·')[1])] # remove the initial space
                    else:
                        dict_events[str(candidate)][year][month][day][columnname] = line[i]
                
                
# In[]               
"""
--- save data ---

"""

with open(os.path.join(path_events, 'Events.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_events, outfile, ensure_ascii=False)                 
                
                
                
                
                
            
