# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:16:02 2019

@author: Yichen Jiang
"""
"""
--- This file is for importing new events data (with more nationwide events for each candidate) ---

"""

# In[]
import csv
import json
import os

# In[]
createVar = locals()

# In[]
# dictionary of month-check
dict_month = {
        'January': '01', \
        'February': '02', \
        'March': '03', \
        'April': '04', \
        'May':'05', \
        'June': '06', \
        'July': '07', \
        'August': '08', \
        'September': '09', \
        'October': '10', \
        'November': '11', \
        'December': '12' \
        }

# In[]
# list of candidates
list_candidates = ['BIDEN', 'HARRIS', 'WARREN', 'SANDERS']

# In[]:
"""
--- import data ---

"""
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science'
path_events = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Events'

filenames = os.listdir(path_events)
filenames_state = os.listdir(path)

# In[]
"""
--- import state information ---

"""

list_state = []
list_statename = []
count = 0

for filename in filenames_state:
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

# In[]
"""
--- import old events data ---

"""
dict_events_old = {}

for filename in filenames:
    if filename.endswith('.json'):
        file = open(os.path.join(path_events, filename), 'r', encoding = 'utf-8')
        for line in file.readlines():
            dict_events_old = json.loads(line)
            
# In[]
"""
--- import new events data and sort ---

"""
# new events data
dict_events_new = {}

# list of columnnames
list_columnnames = []

for filename in filenames:
    if filename.endswith('csv') and 'New' in filename:
        for candidate in list_candidates:
            if candidate in str.upper(filename):
                dict_events_new[candidate] = {}
                break
        count = 0
        file = open(os.path.join(path_events, filename), 'r',encoding='utf-8-sig')
        lines = csv.reader(file)
        for line in lines:
            # 1st row with columnnames
            if count == 0:
                list_columnnames = line
                index_event = list_columnnames.index('event')
                count += 1
            else:
                # extract time of the event
                time = line[index_event].split(':')[0]
                year = time.split(' ')[2]
                month = dict_month[time.split(' ')[0]]
                day = time.split(',')[0].split(' ')[1]
                if int(day) < 10:
                    day = '0'+day
                temp_time = year+month+day
                if temp_time not in dict_events_new[candidate].keys():
                    dict_events_new[candidate][temp_time] = {}
                # save event
                for i in range(len(line)):
                    columnname = list_columnnames[i]
                    dict_events_new[candidate][temp_time][columnname] = line[i]

# In[]
"""
--- combine the events' information ---

"""
dict_events = {}
# 1st: old events data
for candidate in dict_events_new.keys():
    if candidate not in dict_events.keys():
        dict_events[candidate] = {}
    for candidate_old in dict_events_old.keys():
        if str.upper(candidate_old) == candidate:
            for year in dict_events_old[candidate_old].keys():
                for month in dict_events_old[candidate_old][year].keys():
                    # since current data stops on 20190630:
                    if int(month) >= 7:
                        continue
                    for day in dict_events_old[candidate_old][year][month].keys():
                        # save the time of current date into str
                        temp_time = str(year)
                        if int(month) < 10:
                            temp_time += '0'+str(month)
                        else: temp_time += str(month)
                        if int(day) < 10:
                            temp_time += '0'+str(day)
                        else: temp_time += str(day)
            
                        if temp_time not in dict_events[candidate].keys():
                            dict_events[candidate][temp_time] = {}
                        dict_events[candidate][temp_time]['event'] = \
                        dict_events_old[candidate_old][year][month][day]['Event']
                        # check event location
                        location = dict_events_old[candidate_old][year][month][day]['Location']
                        for state in list_state:
                            if state in location:
                                dict_events[candidate][temp_time]['location'] = state
                                dict_events[candidate][temp_time]['event_type'] = 'local'

# 2nd: new events data
for candidate in dict_events_new.keys():
    for date in dict_events_new[candidate].keys():
        if date not in dict_events[candidate].keys():
            dict_events[candidate][date] = {}
        dict_events[candidate][date]['event'] = dict_events_new[candidate][date]['event']
        dict_events[candidate][date]['event_type'] = dict_events_new[candidate][date]['event_type']
        dict_events[candidate][date]['location'] = dict_events_new[candidate][date]['location']
        
# In[]
"""
--- save events ---

"""

with open(os.path.join(path_events, 'New_Events.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_events, outfile, ensure_ascii=False)   
            
            
            
            
            
            
            
            
            
