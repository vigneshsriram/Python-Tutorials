
# coding: utf-8

# In[1]:

# Structure of your solution to Assignment 1 

# add your import statement
import nltk
import re
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
        


    


# In[3]:

class reviewPredictor(object):
    #rp = reviewPredictor("finding_dory_reivew.csv")
    def __init__(self, input_file):
        self.input_file = input_file
    
    #formats the string ie removes all special characters except '=' and '.'
    #and gets all alphabets in the lower case
    def tokenize(self,text):
        text = text.lower()
        text1 = re.findall(r'[a-z\.\-]',text)
        #print text1
        pattern=r'[a-z]+[a-z\-\.]*'
        tokens=nltk.regexp_tokenize(text, pattern)
        # print(tokens)
        return tokens
    
    
#     determines the sentiment of the string as follows:
#     if the number of positive words > the number of negative words, the sentiment is positive
#     if the number of positive words < the number of negative words, the sentiment is negative
#     if the number of positive words = the number of negative words, the sentiment is neutral
    def sentiment_analysis(text):
        WrongReview = 0;
        RightReview = 0;
        with open("finding_dory_reivew.csv") as f:
            tokens = f.readlines()
            print("Total Reviews "+str(len(tokens)))
        
        with open('positive-words.txt') as f:
        
            positiveWords = [line.strip() for line in f]
        
        with open('negative-words.txt') as f:
        
            negativeWords = [line.strip() for line in f]
            
        for i in (tokens):
            
            negativeCounter = 0;
            positiveCounter = 0;
            
            ti = i[::-1]
            it=""
            for j in ti:
                if j == ",":
                    break;
                else:
                    it = it+j
            it = it[::-1]
            
            it = it.strip()
            #print(i)
            i=i[:-len(it)-1]
            
            tokenList = rp.tokenize(i)
            
            filtered_words =[]
            for word in tokenList:
                if word not in stop_words:
                    filtered_words.append(word)
            
            for word in filtered_words:
                if word in negativeWords:
                    negativeCounter = negativeCounter+1
            for word in filtered_words:
                if word in positiveWords:
                    positiveCounter = positiveCounter+1
            
            
            if((negativeCounter>positiveCounter and it == "negative") or                (negativeCounter<positiveCounter and it == "positive") or                (negativeCounter==positiveCounter and it == "neural ")):
                RightReview = RightReview +1
            else:
                WrongReview = WrongReview+1
                
            
        #To display the results
        print("The number of Wrong Predictions are " +str(WrongReview))
        print("The number of Right Predictions are " +str(RightReview))   
        
        return (len(tokens),RightReview,WrongReview)
if __name__ == "__main__":  
    
    #We load the .csv file
    rp = reviewPredictor("finding_dory_reivew.csv")
    #call the method sentiment_analysis() and save the returned answer in a list
    list1 = rp.sentiment_analysis()
    print(list1)
    #calculating the right predictions correctlyMarked/Total
    a = float(list1[1])/float(list1[0])
    print("The Right predcitions are " +str(a))
    


    


# In[ ]:



