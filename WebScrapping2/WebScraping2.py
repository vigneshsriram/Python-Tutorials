
# coding: utf-8

# In[1]:

#define a listener which listens to tweets in real time


import tweepy
# to install tweepy, use: pip install tweepy

# import twitter authentication module
from tweepy import OAuthHandler

# import tweepy steam module
from tweepy import Stream

# import stream listener
from tweepy.streaming import StreamListener

# import the python package to handle datetime
import datetime

# set your keys to access tweets 
# you can find your keys in Twitter.
consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
# Customize a tweet event listener 
# inherited from StreamListener provided by tweepy
# This listener reacts when a tweet arrives or an error happens

class MyListener(StreamListener):
    
    # constructor
    def __init__(self, output_file, time_limit):
        
            # attribute to get listener start time
            self.start_time=datetime.datetime.now()
            
            # attribute to set time limit for listening
            self.time_limit=time_limit
            
            # attribute to set the output file
            self.output_file=output_file
            
            # initiate superclass's constructor
            StreamListener.__init__(self)
    
    # on_data is invoked when a tweet comes in
    # overwrite this method inheritted from superclass
    # when a tweet comes in, the tweet is passed as "data"
    def on_data(self, data):
        
        # get running time
        running_time=datetime.datetime.now()-self.start_time
        print(running_time)
        
        # check if running time is over time_limit
        if running_time.seconds/60.0<self.time_limit:
            
            # ***Exception handling*** 
            # If an error is encountered, 
            # a try block code execution is stopped and transferred
            # down to the except block. 
            # If there is no error, "except" block is ignored
            try:
                # open file in "append" mode
                with open(self.output_file, 'a') as f:
                    # Write tweet string (in JSON format) into a file
                    f.write(data)
                    
                    # continue listening
                    return True
                
            # if an error is encountered
            # print out the error message and continue listening
            
            except BaseException as e:
                print("Error on_data:" , str(e))
                
                # if return "True", the listener continues
                return True
            
        else:  # timeout, return False to stop the listener
            print("time out")
            return False
 
    # on_error is invoked if there is anything wrong with the listener
    # error status is passed to this method
    def on_error(self, status):
        print(status)
        # continue listening by "return True"
        return True


# In[ ]:

# Collect tweets with specific topics within 2 minute

# initiate an instance of MyListener 
tweet_listener=MyListener(output_file="srksalman.txt",time_limit=10)

# start a staeam instance using authentication and the listener
twitter_stream = Stream(auth, tweet_listener)
# filtering tweets by topics
twitter_stream.filter(track=['#SlapAFilm', '#ISurviveTwitterBy','Kylie Jenner'])


# In[ ]:

tweet_listener=MyListener(output_file="newsrksalman.txt",time_limit=10)
twitter_stream = Stream(auth, tweet_listener)
twitter_stream.filter(track=['#SlapAFilm', '#ISurviveTwitterBy','Kylie Jenner'])
#twitter_stream.sample()


# In[14]:

#Read/write JSON 
import json
tweets=[]

with open('newsrksalman.txt', 'r') as f:
    # each line is one tweet string in JSON format
    for line in f: 
        
        # load a string in JSON format as Python dictionary
        tweet = json.loads(line) 
              
        tweets.append(tweet)

# write the whole list back to JSON
json.dump(tweets, open("all_tweets.json",'w'))

# to load the whole list
# pay attention to json.load and json.loads
tweets=json.load(open("all_tweets.json",'r'))


# In[42]:

# A tweet is a dictionary
# Some values are dictionaries too!
# for details, check https://dev.twitter.com/overview/api/tweets

print("# of tweets:", len(tweets))
first_tweet=tweets[0]

print("\nprint out first tweet nicely:")
print(json.dumps(first_tweet, indent=4)) 
print (tweets[0]["text"])



# In[51]:


print(len(tweets))
text = ""
for i in range(0,400):
    text = text + tweets[i]["text"] 
    
print (text)



# In[55]:

noUnicode = text.encode('utf8')
print(type(noUnicode))
print(noUnicode)


# In[71]:

text = text.replace(",","").lower()
a = text.split(" ")

print(a)


# In[72]:

for i,t in enumerate(a):
    a[i]=a[i].encode('utf8')
    print(type(a[i]))
    


# In[73]:

count_per_topic={}

for word in a:
    if word in count_per_topic:
        count_per_topic[word]+=1
    else:
        count_per_topic[word]=1
    
print(count_per_topic)


# In[74]:

sorted_topics = sorted(count_per_topic.items(),key=lambda item:-item[1])
print(sorted_topics)


# In[75]:

top_50_topics=sorted_topics[0:50]
print(top_50_topics)


# In[76]:

topics,count = zip(*top_50_topics)
print(topics,count)


# In[77]:

import pandas as pd
import brunel

df = pd.DataFrame(top_50_topics,columns=["topic","count"])

get_ipython().magic(u"brunel data('df') label(topic) size(count) color(topic) bubble sort(count)tooltip(count)")


# In[ ]:



