# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:43:21 2019

@author: Yichen Jiang
"""
"""
This file is for automatically collecting Andrew Yang's tweets

"""
# In[]
"""
1) import packages
"""



import json
import os
import tweepy
import re
import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from pytz import timezone
import logging
import OpenSSL


# In[]
"""
2) define saving_directory
"""



path_local = r'C:\Users'


# In[]
"""
3) import keys and secrets for API

basically api_1 is for streaming, and api_2 is for re-collecting tweets by using tweet_id
"""



"""
--- my first key and secret ---
​
"""
consumer_key_1 = ''  
consumer_secret_1 = ''  
access_token_key_1 = ''  
access_token_secret_1 = ''  


"""
--- my second key and secret ---
​
"""
consumer_key_2 = ''  
consumer_secret_2 = ''  
access_token_key_2 = ''  
access_token_secret_2 = '' 


"""
--- API ---
​
"""
auth_1 = tweepy.OAuthHandler(consumer_key_1, consumer_secret_1)  
auth_1.set_access_token(access_token_key_1, access_token_secret_1)  

api_1 = tweepy.API(auth_1, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


auth_2 = tweepy.OAuthHandler(consumer_key_2, consumer_secret_2)  
auth_2.set_access_token(access_token_key_2, access_token_secret_2)  

api_2 = tweepy.API(auth_2, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[]
"""
4) setup dictionaries, lists and formats
"""



# create dictionary
dict_ay = {}
list_id = []
list_api = [api_1,api_2]
# time format
fmt_twitter = '%a %b %d %H:%M:%S %z %Y'
fmt_std = '%Y-%m-%d %H:%M:%S %Z%z'
fmt_rt = '%Y-%m-%d %H:%M:%S'



# In[]
"""
5) define functions
"""




# function for collecting tweets
def get_tweets(tweet_id, API):
    global dict_ay
    for tweet_key in dict_ay.keys():
        if tweet_id in tweet_key:
            try:
                api = API
                # get the first collecting time of current tweet
                initial_key = list(dict_ay[tweet_key]['original_tweets'].keys())[0]
                # current timestamp -- collecting time in PST timezone
                curr_time = datetime.now(timezone('US/Pacific')).strftime(fmt_std)
                # track tweet by using the second api
                tweet = api.get_status(id = tweet_id, tweet_mode = 'extended')
                dict_ay[tweet_key]['original_tweets'][curr_time] = {}
                for attribute in tweet._json:
                    if attribute not in dict_ay[tweet_key]['original_tweets'][initial_key].keys() or \
                    tweet._json[attribute] != dict_ay[tweet_key]['original_tweets'][initial_key][attribute]: # tweet attributes may be updated
                        dict_ay[tweet_key]['original_tweets'][curr_time][attribute] = tweet._json[attribute]
            except tweepy.TweepError as e:
                if e.api_code == 144:
                    print('tweet_id:',tweet_id,' has been deleted by AndrewYang or Original User!')
                    # then remove this job
                    scheduler.remove_job(job_id = tweet_id)
                    print('job_id:',tweet_id,' has been removed!')
                else:
                    print('error code is:', e.api_code)
            except Exception as e:
                print(e)
                
                
#------------------------------------------------------------------------------------------------------------------
# function for saving tweets
def save_tweets():
    global path_local
    global path_rivanna
    global dict_ay
    with open(os.path.join(path_local, 'dict_ay.json'), 'w+', encoding="utf-8") as outfile:
        json.dump(dict_ay, outfile, ensure_ascii=False) 
    curr_time = datetime.now(timezone('US/Eastern')).strftime(fmt_std)
    print(curr_time, ' : tweets saved')
    
#------------------------------------------------------------------------------------------------------------------
# functino for changing collecting time​
def change_time():
    global dict_ay
    global scheduler
    
    for job in scheduler.get_jobs():
        # check if current job is a tracking job
        if bool(re.search(r'\d', job.id)) == True:
            # check tweet
            for tweet_key in dict_ay.keys():
                if job.id in tweet_key:
                    # check if the cycle of current should been modified but not yet
                    # get the first collecting time of current tweet
                    tweet_time = datetime.strptime(tweet_key[0:19],fmt_rt)
                    curr_time = datetime.strptime(datetime.now(timezone('US/Pacific')).strftime(fmt_rt),fmt_rt)
                    # 1st modification
                    if (curr_time-tweet_time).days >= 1 and str(job.trigger) == 'interval[1:00:00]':
                        temp_trigger = scheduler._create_trigger(trigger='interval',trigger_args={'hours':4})
                        # new starttime == old starttime + 3hs (4h - 1h)
                        next_run_time = job.trigger.start_date+timedelta(hours=3)
                        scheduler.modify_job(job.id, trigger=temp_trigger, next_run_time=next_run_time)
                        dict_ay[tweet_key]['marker'] = 'tweet is collected every 4 hours'
                        print('tweet: ',job.id,' has been changed to be collected every 4 hours')
                    # 2nd modification
                    # 1h*24 + 4h*6 + 4h*6 = 3 days
                    # if more than 3 days
                    elif (curr_time-tweet_time).days >= 3 and str(job.trigger) == 'interval[4:00:00]':
                        temp_trigger = scheduler._create_trigger(trigger='interval',trigger_args={'days':1})
                        # new starttime == old starttime + 1hs (1day - 4h)
                        next_run_time = job.trigger.start_date+timedelta(hours=20)
                        scheduler.modify_job(job.id, trigger=temp_trigger, next_run_time=next_run_time)
                        dict_ay[tweet_key]['marker'] = 'tweet is collected every 1 day'
                        print('tweet: ',job.id,' has been changed to be collected every 1 day')
                    # if more than 2 weeks, then stop collecting tweets and delete job
                    if (curr_time-tweet_time).days >= 14:
                        try:
                            scheduler.remove_job(job.id)
                            dict_ay[tweet_key]['marker'] = 'tweet has been stopped to be collected'
                            print('tweet: ', job.id,' has been stopped to be collected.')
                        except Exception as e:
                            print(e)
                            
#------------------------------------------------------------------------------------------------------------------
# function for checking api limit and changing api​
def check_limit():
    global scheduler
    global list_api
    list_goodapi = []
    # 1st, check api limits
    for api in list_api:
        dict_limit = api.rate_limit_status() 
        for rate_type in dict_limit['resources'].keys():
            for sub_rate_type in dict_limit['resources'][rate_type].keys():
                if dict_limit['resources'][rate_type][sub_rate_type]['limit'] != dict_limit['resources'][rate_type][sub_rate_type]['remaining']:
                    if dict_limit['resources'][rate_type][sub_rate_type]['remaining']/dict_limit['resources'][rate_type][sub_rate_type]['limit'] \
                    <= 0.1: # meet rate limit
                        continue
                    else: list_goodapi.append(api)
    # check if any api doesn't meet rate limit
    if len(list_goodapi) == len(list_api) or len(list_goodapi) == 0:
        # all apis are good or bad, keep going
        pass
    else: # not all api are good
        # choose the first api in the list_goodapi
        api = list_goodapi[0]
        # check each job's api
        for job in scheduler.get_jobs():
            # check if current job is a one-hour tracking job
            if bool(re.search(r'\d', job.id)) == True and str(job.trigger) == 'interval[1:00:00]':
                if job.args[1] != api:
                    scheduler.modify_job(job.id, args=[job.id,api])



# In[]
"""
6) define tweepy streaming class
"""




class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
    def on_connect(self):
        print('Sucessfully connect to streaming server')
        pass
    def on_status(self, status):
        global dict_ay
        global list_id
        global scheduler
        # current timestamp -- collecting time in PST timezone
        curr_time = datetime.now(timezone('US/Pacific')).strftime(fmt_std)
        # 1st, check if tweet from Andrew Yang 
        if status._json['user']['id_str'] == '2228878592':
            tweet_time = \
            datetime.strptime(status._json['created_at'], fmt_twitter).astimezone(timezone('US/Pacific')).strftime(fmt_std)
            tweet_id = status.id_str
            tweet_key = tweet_time+'|'+tweet_id
            if tweet_key not in dict_ay.keys():
                dict_ay[tweet_key] = {'original_tweets':{},
                       'replies':{'reply_count':0,
                                  'tweet_id':[]},
                       'quotes':{'quote_count':0,
                                 'tweet_id':[]},
                       'retweets':{'retweet_count':0,
                                   'tweet_id':[]},
                       'marker': 'tweet is collected every 1 hour'}
            # save tweet
            dict_ay[tweet_key]['original_tweets'][curr_time] = status._json
            print(status.text)
            # add job
            initial_run_time = datetime.now()+timedelta(hours=1)
            starting_time = initial_run_time.strftime(fmt_rt)
            list_args = []
            list_args.append(tweet_id)
            list_args.append(api_2) # initially assign api_2 for tracking tweets
            try:
                scheduler.add_job(get_tweets, 'interval', id=tweet_id, args = list_args, hours=1, max_instances=10, next_run_time=initial_run_time, start_date=starting_time, end_date='2022-12-31 23:59:00')
            except Exception as e:
                print(e)
            # save tweet_id into list_id
            list_id.append(tweet_id)
            
        # 2nd, check if tweet is a reply to one of Yang's tweets
        if status._json['in_reply_to_status_id_str'] in list_id:
            tweet_id = status.id_str
            # find that tweet and save this reply
            for tweet_key in dict_ay.keys():
                if status._json['in_reply_to_status_id_str'] in tweet_key:
                    dict_ay[tweet_key]['replies']['reply_count'] += 1
                    dict_ay[tweet_key]['replies']['tweet_id'].append(tweet_id)
                    
        # 3rd, check if tweet is a quote of one of Yang's tweets
        if 'quoted_status_id_str' in status._json.keys() and status._json['quoted_status_id_str'] in list_id:
            tweet_id = status.id_str
            # find that tweet and save this quote
            for tweet_key in dict_ay.keys():
                if status._json['quoted_status_id_str'] in tweet_key:
                    dict_ay[tweet_key]['quotes']['quote_count'] += 1
                    dict_ay[tweet_key]['quotes']['tweet_id'].append(tweet_id)
                    
        # 4th, check if tweet is a retweet of one of Yang's tweets
        if 'retweeted_status' in status._json.keys() and status._json['retweeted_status']['id_str'] in list_id:
            tweet_id = status.id_str
            # find that tweet and save this quote
            for tweet_key in dict_ay.keys():
                if status._json['retweeted_status']['id_str'] in tweet_key:
                    dict_ay[tweet_key]['retweets']['retweet_count'] += 1
                    dict_ay[tweet_key]['retweets']['tweet_id'].append(tweet_id)
        
    def on_exception(self, exception):
        print(exception)
        return False
    def on_error(self, status_code):
        if status_code == 420:
            time.sleep(200)
            #returning False in on_error disconnects the stream
            return False
    def on_timeout(self):
        myStream.filter(follow=['2228878592'],is_async=True)
        
    def on_disconnect(self, notice):
        print(notice)
        myStream.filter(follow=['2228878592'],is_async=True)



# In[]
"""
7) create scheduler and start running jobs
"""




scheduler = BackgroundScheduler()
scheduler.add_job(save_tweets,'interval', id='save_tweet', hours=2, start_date='2019-11-30 00:00:00', end_date='2022-12-31 23:59:59')
scheduler.add_job(change_time,'interval', id='change_time', hours=1, start_date='2019-11-30 00:00:00', end_date='2022-12-31 23:59:59')
scheduler.add_job(check_limit,'interval', id='check_limit', minutes=15, start_date='2019-11-30 00:00:00', end_date='2022-12-31 23:59:59')
scheduler.start()



# In[]
"""
8) connect to the stream (run this cell again to reconnect to internet if lose connect)

tweepy streaming cannot be shutdown immediately, please try restarting the kernel 
to stop the streaming if necessary. 
"""



try:
    if __name__ == '__main__':
        # step 2: creating a stream
        myStreamListener = MyStreamListener()
        myStream = tweepy.Stream(auth = api_1.auth, listener=myStreamListener)
        # step 3: starting a stream
        myStream.filter(follow=['2228878592'],is_async=True) # AndrewYang's user_id
# various exception handling blocks
except ConnectionError as e:
    print(e)
    myStream.filter(follow=['2228878592'],is_async=True)
except OpenSSL.SSL.WantReadError as e:
    logging.error('WantReadError: %s', e)
except tweepy.TweepError as e:
    print('Below is the printed exception')
    print(e)
    if '401' in e:    
        # not sure if this will even work
        print('Below is the response that came in')
        print(e)
        time.sleep(60)
    else:
        #raise an exception if another status code was returned
        raise e
except Exception as e:
    print('Unhandled exception')
    print(e)



# In[]
"""
9) check rate limit
"""

API = api_1 # api_1 or api_2

dict_limit = API.rate_limit_status() 
for rate_type in dict_limit['resources'].keys():
    for sub_rate_type in dict_limit['resources'][rate_type].keys():
        if dict_limit['resources'][rate_type][sub_rate_type]['limit'] != dict_limit['resources'][rate_type][sub_rate_type]['remaining']:
            print('rate_type: ', rate_type, 'sub_rate_type: ', sub_rate_type)
            print(rate_type, 'limit:', dict_limit['resources'][rate_type][sub_rate_type]['limit'], 'remaining:', dict_limit['resources'][rate_type][sub_rate_type]['remaining'])
            print('\n')

# In[]
tweets = API.user_timeline(count = 20)
# In[]
list_tweetid = []
for tweet in tweets:
    list_tweetid.append(tweet._json['id_str'])

# In[]
    
dict_tweets = {}

recollected_tweets = api_1.statuses_lookup(list_tweetid) # id_list is the list of tweet ids
for tweet in recollected_tweets:
    dict_tweets[tweet._json['id_str']] = tweet._json
    print(tweet._json['id_str'],' has been collected at ', datetime.now())
    


# In[]
"""
10) check all the jobs we have on the scheduler
"""



scheduler.print_jobs()



# In[]
"""
【cells below are unnecessary. 】
"""
# In[]
"""
shutdown the streaming (tweepy streaming cannot be shutdown immediately, 
please try restarting the kernel to stop the streaming if necessary)
"""



myStream.disconnect()



# In[]
"""
shutdown the scheduler if necessary
"""



scheduler.shutdown()



# In[]
"""
manually save the tweets
"""



with open(os.path.join(path_local, 'dict_ay.json'), 'w+', encoding="utf-8") as outfile:
        json.dump(dict_ay, outfile, ensure_ascii=False) 



# In[]
"""
manually import tweets
"""



with open(os.path.join(path_local, 'dict_ay.json'), 'r', encoding="utf-8") as file:
    for line in file.readlines():
        dict_ay = json.loads(line)



# In[]
"""
manually get list_id
"""



list_id = []
for tweet_key in dict_ay.keys():
    list_id.append(tweet_key.split('|')[1])



# In[]
"""
check tweets
"""



for tweet_key in dict_ay.keys():
    print(tweet_key)
