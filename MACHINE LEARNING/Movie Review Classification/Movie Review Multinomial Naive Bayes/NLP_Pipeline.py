#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


from nltk.corpus.reader import reviews
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys

# In[12]:


tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


# In[13]:


def getCleanReview(review):
    review = review.lower()
    review = review.replace("<br /><br />"," ") # replacing break with space
    
    #Tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    cleaned_review = ' '.join(stemmed_tokens)
    
    return cleaned_review




# In[ ]:
# Doing input file and returning a clean output file of moview reviews
# def getStemmedDocument(inputFile,outputFile):

#     out = open(outputFile,'w',encoding="utf8")

#     with open(inputFile,encoding="utf8") as f:
#         reviews=f.readlines()

#     for review in reviews:
#         cleaned_review = getCleanReview(review)
#         print((cleaned_review),file = out)

#     out.close()


# # Reading command line input
# inputFile = sys.argv[1]
# outputFile = sys.argv[2]
# getStemmedDocument(inputFile,outputFile)

