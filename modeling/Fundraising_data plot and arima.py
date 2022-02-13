# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:12:27 2019

@author: Yichen Jiang
"""

"""
--- this file is for plotting fundraising data with new events data ---

"""

# In[]
import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyflux as pf
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
--- import events data ---

"""

for filename in filenames_events:
    if filename.endswith('.json') and 'New' in filename:
        file = open(os.path.join(path_events, filename), 'r', encoding = 'utf-8')
        for line in file.readlines():
            dict_events = json.loads(line)

# In[]
"""
--- read dict_data file ---

"""
for filename in filenames_finance:
    if 'dict_data' in filename:
        file = open(os.path.join(path_finance, filename), 'r', encoding = 'utf-8')
        for line in file.readlines():
            dict_data = json.loads(line)

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
# list of months
list_month = list(dict_month.items())

# In[]

"""
--- sort data for plot ---

"""

dict_plot = {}

for candidate in dict_events.keys():
    if candidate not in dict_plot.keys():
        dict_plot[candidate] = {}
        # variables from dict_data (fundraising data)
        list_temp = sorted(dict_data['finance'][candidate].items(),key=lambda x:x[0],reverse=False)
        for line in list_temp:
            for variable in line[1].keys():
                if variable not in dict_plot[candidate].keys():
                    dict_plot[candidate][variable] = []
                dict_plot[candidate][variable].append(line[1][variable])

# In[]
# variables from dict_events (events data)
# add month column as well
past_month = ''
current_month = ''

for candidate in dict_events.keys():
    dict_plot[candidate]['local'] = []
    dict_plot[candidate]['nationwide'] = []
    dict_plot[candidate]['month'] = []
    list_temp = sorted(dict_data['finance'][candidate].items(),key=lambda x:x[0],reverse=False)
    for line in list_temp:
        date = line[0]
        # no event on current date
        if date not in dict_events[candidate].keys():
            dict_plot[candidate]['local'].append(0)
            dict_plot[candidate]['nationwide'].append(0)
        else:
            if dict_events[candidate][date]['event_type'] == 'nationwide':
                dict_plot[candidate]['local'].append(0)
                dict_plot[candidate]['nationwide'].append(1)
            elif dict_events[candidate][date]['event_type'] == 'local':
                dict_plot[candidate]['local'].append(1)
                dict_plot[candidate]['nationwide'].append(0)                 
        # check month
        for pair in list_month:
            if date[4:6] == pair[1]:
                current_month = pair[0]
                break
        if current_month != past_month:
            dict_plot[candidate]['month'].append(current_month)
            past_month = current_month
        else:
            dict_plot[candidate]['month'].append('')
        
        
                
# In[]
"""
--- generate figures ---

"""
"""
--- plot values ---
--- new donors ---
--- amount of new donations ---
"""
# dictionary of font size
dict_size = {67: 18, 132: 16, 142: 16, 161: 14}

for candidate in dict_plot.keys():
    r = 'k'
    for variable in dict_plot[candidate].keys():
        if variable == 'number_of_new_donors' or variable == 'amount_of_new_donations':
            # plot
            length = len(dict_plot[candidate]['month'])
            #font_size = dict_size[length]
            plt.style.use('ggplot')
            fig = plt.figure(dpi = 80, figsize = (0.5*length, 0.25*length))
            plt.title('Candidate: '+str(candidate)+ ','+str(variable), fontdict={'size':'60'})
            plt.subplot(1, 1, 1)
            
            values = dict_plot[candidate][variable]
            index = np.arange(length)
            plt.plot(index, values, label=str(variable), color=r)       

            # nationwide events & local events
            
            for i in range(length):
                if dict_plot[candidate]['local'][i] == 1:
                    plt.axvline(i, color='black', linestyle='--')
                if dict_plot[candidate]['nationwide'][i] == 1:
                    plt.axvline(i, color='red', linestyle='--')
            local = plt.axvline(i, color='black', linestyle='--', label = 'Local events')
            nationwide = plt.axvline(i, color='red', linestyle='--', label='Nationwide events')
            # local events
            #for i in range(length):
            #    marker = dict_plot[candidate]['local']
            #    plt.text(index[i], values[i], str(marker[i]), linestyle='|', color=r)       
            
            plt.xlabel('Date', fontdict={'size':'60'})
            plt.ylabel(str(variable), fontdict={'size':'60'})
            plt.xticks(index, dict_plot[candidate]['month'], fontsize=60)
            #plt.xticks(index, dict_plot[candidate]['time'], fontsize=font_size, rotation = 45)
            plt.yticks(fontsize = 60)
            plt.legend(handles = [local, nationwide], loc='upper right',fontsize = 60)
            plt.show()    

# In[]
"""
--- fit with ARIMAX model with intervention ---

"""
# create dataframe
for candidate in dict_plot.keys():
    createVar['df_'+str(candidate)] = pd.DataFrame.from_dict(dict_plot[candidate])

# In[]
"""
--- standardize the data ---

"""
for candidate in dict_plot.keys():
    for i in createVar['df_'+str(candidate)].index:
        for column in createVar['df_'+str(candidate)].columns:
            if ('event' in column and 'guests' not in column) or column == 'month':
                continue
            createVar['df_'+str(candidate)][column] = \
            (createVar['df_'+str(candidate)][column]-createVar['df_'+str(candidate)][column].min())/\
            (createVar['df_'+str(candidate)][column].max()-createVar['df_'+str(candidate)][column].min())
        break
    
# In[]
# create model
        
candidate = 'BIDEN'
dependent_variable = 'amount_of_new_donations'


model = dependent_variable+'~1+local+nationwide'

for variable in dict_plot[str(candidate)].keys():
    if 'event' in variable and str(candidate) not in variable and 'guests' not in variable:
        model += '+'+str(variable)

# In[]

# fit the model
ARIMAX_model = pf.ARIMAX(data=createVar['df_'+str(candidate)], \
formula=model,ar=2, integ=0, ma=2,family=pf.Normal())
ARIMAX_result = ARIMAX_model.fit("MLE")
ARIMAX_result.summary()




