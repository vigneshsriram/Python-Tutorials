
# coding: utf-8

# In[34]:

import nltk, re, string
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
# numpy is the package for matrix cacluation
import numpy as np  
# for lemma
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
wordnet_lemmatizer = WordNetLemmatizer()

list1 = []
import csv
with open('amazon_review_300.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        list1.append(row[2])
print(list1[0])
        
#Tokenize the documents
def get_doc_tokens(doc):
    lemmatized = false;
    stop_words = stopwords.words('english')
    
    #condition to lemmatize
    if(lemmatized == True):
        tokens=[token.strip()             
        for token in nltk.word_tokenize(doc.lower())             
            if token.strip() not in stop_words and               
            token.strip() not in string.punctuation]
        #tokens = lemma(doc)
        tagged_tokens= nltk.pos_tag(tokens)
        #print("********************TAgged Tokens***********************")
        #print(tagged_tokens)
        le_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag))           
        # tagged_tokens is a list of tuples (word, tag)
        for (word, tag) in tagged_tokens \
        # remove stop words
        if word not in stop_words and \
        # remove punctuations
        word not in string.punctuation]
        # get lemmatized unique tokens as vocabulary
        le_vocabulary=set(le_words)
        tokens = list(le_vocabulary)
    
    else:
        #stop_words = stopwords.words('english')
        tokens=[token.strip()             
                for token in nltk.word_tokenize(doc.lower())             
                if token.strip() not in stop_words and               
                token.strip() not in string.punctuation]
    # you can add bigrams, collocations, or lemmatization here
    #print("******************** Tokens***********************")
    return tokens

def get_wordnet_pos(pos_tag): # 'JJ','NN'
    
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ
    
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB
    
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN

    
        
def tfidf(list1):
    # step 2. process all documents to get list of token list
    docs_tokens=[get_doc_tokens(doc) for doc in list1]
    #print(docs_tokens)
    voc=list(set([token for tokens in docs_tokens               
                  for token in tokens]))
    dtm=np.zeros((len(list1), len(voc)))
    #print(voc)
   
    
    for row_index,tokens in enumerate(docs_tokens):
        for token in tokens:
            col_index=voc.index(token)
            dtm[row_index, col_index]+=1
            #print(row_index , col_index , dtm[row_index, col_index])
    print("*********************Length of the Matrix*****************************")
    print(dtm.shape)
       
    # step 4. get normalized term frequency (tf) matrix        
    doc_len=dtm.sum(axis=1, keepdims=True)
    tf=np.divide(dtm, doc_len)
    
    
    # step 5. get idf
    doc_freq=np.copy(dtm)
    doc_freq[np.where(doc_freq>0)]=1

    smoothed_idf=np.log(np.divide(len(list1)+1, np.sum(doc_freq, axis=0)+1))+1

    
    # step 6. get tf-idf
    smoothed_tf_idf=normalize(tf*smoothed_idf)
    return smoothed_tf_idf

smoothed_tf_idf = tfidf(list1)


from scipy.spatial import distance
similarity=1-distance.squareform(distance.pdist(smoothed_tf_idf, 'cosine'))

docs_similar = similarity[0].tolist()
similarList = sorted(enumerate(docs_similar), key=lambda item:-item[1])[:11]
print("***********The most similar reviews are*************")
for i in similarList:
    print(i[0] , list1[i[0]])
    print("*******************************************************************************")
print(similarList)


# In[ ]:



