#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import os
import codecs


# In[127]:


import nltk
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
import time
from  nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


# In[128]:


from num2words import num2words


# In[129]:


import time


# In[130]:


def magnitude(array_list):
    s=0
    for i in range(len(array_list)):
        s=s+(array_list[i]**2)
    return np.power(s,0.5)


# In[131]:


#geeks for geeks 
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z 


# In[132]:


import copy
from nltk.stem.wordnet import WordNetLemmatizer


# In[133]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())


# In[134]:


def punc_improve(str1):
    s=[]
    s1=''
    for i in range(len(str1)):
        s.append(str1[i])
#     print(s)

    for i in range(len(s)):
        if (s[i]=='.' or s[i]==',' or  s[i]=='!' or  s[i]=='*' or s[i]=='+'  or s[i]=='-' or s[i]=='\"'  or s[i]=='\'' or
            s[i]=='{' or s[i]=='}' or s[i]==';' or s[i]==':' or s[i]=='(' or s[i]==')' or
        s[i]=='='  or s[i]=='@' or s[i]=='>' or s[i]=='[' or s[i]==']' or s[i]=='|' or s[i]=='/'
        or s[i]=='#' or s[i]=='%' or s[i]=='`' or s[i]=='~' or s[i]=="/" or s[i]=='_' or s[i]=='<' or s[i]=='?' or  s[i]==' ' or s[i]=='$' or s[i]=='^'or s[i]=='' or s[i]==' ' or s[i]=='&'):
            pass
        else:
            s1=s1+s[i]
        
    return s1


# In[135]:


def dash_improve(str1):
    s=[]
    s1=''
    for i in range(len(str1)):
        s.append(str1[i])
#     print(s)

    for i in range(len(s)):
        if s[i]!='-':
            s1=s1+s[i]
    return s1


# In[136]:


start1=time.time()


# In[137]:


os.chdir('stories')


# In[138]:


f=open('index1.html')
html1=f.read()

soup=BeautifulSoup(html1,features="lxml")
a=[]
e=[]
for link in soup.find_all("a"):
    a.append(link.get("href"))
for link in soup.find_all("table"):
    e.append(link.get("href"))


# In[139]:


table = soup.find_all('table')[0] # Grab the first table

new_table = pd.DataFrame(columns=range(0,2), index = [0]) # I know the size 
row_marker = 0
st=[]
f=[]
for row in table.find_all('tr'):
    column_marker = 0
    columns = row.find_all('td')
    for column in columns:
        st = column.get_text()
        f.append(st)
        column_marker += 1
g=f[0]
line=g.split('\n')
start=[]
text_file1=[]
size_file1=[]
title1=[]
for i in range(len(line)-1):
    start=line[i].split(' ')
    text_file1.append(start[0])
    size_file1.append(start[2])
    temp=start[3:]

    temp1=''

    for j in temp:
        temp1=temp1+" "+j
    
    title1.append(temp1)
    


# In[ ]:





# In[ ]:





# In[140]:


f=open('index.html')
html1=f.read()

soup=BeautifulSoup(html1,features="lxml")
a=[]
e=[]
for link in soup.find_all("a"):
    a.append(link.get("href"))
for link in soup.find_all("table"):
    e.append(link.get("href"))
table = soup.find_all('table')[0] # Grab the first table

new_table = pd.DataFrame(columns=range(0,2), index = [0]) # I know the size 


    


# In[141]:


row_marker = 0
st=[]
f=[]
for row in table.find_all('tr'):
    column_marker = 0
    columns = row.find_all('td')
    for column in columns:
        st = column.get_text()
        f.append(st)
        column_marker += 1


# In[142]:


g=f[7:]
#Till here done


# In[143]:


k=g[0]


# In[144]:


line=k.split('\n')
start=[]
text_file2=[]
size_file2=[]
title2=[]
for i in range(len(line)-1):
    start=line[i].split(' ')
    text_file2.append(start[0])
    size_file2.append(start[2])
    temp=start[3:]

    temp1=''

    for j in temp:
        temp1=temp1+" "+j
    
    title2.append(temp1)


# In[145]:


f=open('index3.html')
html1=f.read()

soup=BeautifulSoup(html1,features="lxml")
a=[]
e=[]
for link in soup.find_all("a"):
    a.append(link.get("href"))
for link in soup.find_all("table"):
    e.append(link.get("href"))
table = soup.find_all('table')[0] # Grab the first table

new_table = pd.DataFrame(columns=range(0,2), index = [0]) # I know the size 
row_marker = 0
st=[]
f=[]
for row in table.find_all('tr'):
    column_marker = 0
    columns = row.find_all('td')
    for column in columns:
        st = column.get_text()
        f.append(st)
        column_marker += 1

    


# In[146]:


start2=time.time()
print(start2-start1)


# In[147]:


text_file3=[]
size_file3=[]
title3=[]
for i in range(len(f)):
    
    if i%3==0:
        text_file3.append(f[i])
    if i%3==1:
        size_file3.append(f[i])
    if i%3==2:
        title3.append(f[i])
        


# In[148]:


text_file=[]
size_file=[]
title=[]
text_file=text_file1+text_file2+text_file3
size_file=size_file1+size_file2+size_file3
title=title1+title2+title3


# In[ ]:





# In[ ]:





# In[149]:


start3=time.time()


# In[150]:


# For normalize the title

final=copy.deepcopy(title)
temp2=[]
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
temp1=[]
for k in range(len(final)):
        temp=[]
        temp=tokenizer.tokenize(final[k])
        temp = copy.deepcopy([w.lower() for w in temp])
        temp=copy.deepcopy([lemmatizer.lemmatize(w,pos='v') for w in temp])
        temp=copy.deepcopy(list(set(temp)-set(stop_words)))
#         print("Temp : ",temp)
#             print("Stop Wro",temp)
        temp1=[]
        for n in range(len(temp)):
    

            t3=str(punc_improve(str(temp[n])))

            temp[n]=t3
            t5=''
            if temp[n].isdigit():
                t5=num2words(int(temp[n]))
            t6=t5.split(' ')
#             print("T6 : ",t6)
            t7=''
            if t6==[""] or t6==['']:
                pass
            else:
                for m in range(len(t6)):
                    t3=str(punc_improve(str(t6[m])))
                    t6[m]=t3
                    t1=str(dash_improve(str(t6[m])))
                    t6[m]=t1
                    t7=t7+t6[m]
                if temp[n]!='':

                    temp[n]=t7
            t1=str(dash_improve(str(temp[n])))
            temp[n]=t1
        if temp==list(['']) or temp==list([' ']) :
            pass
        else:
            temp1=temp1+temp
#             print(temp)
        s=''
        for i in range(len(temp1)):
            s=s+temp1[i]+" "
        temp2.append(s)



# In[151]:


title=copy.deepcopy(temp2)


# In[152]:


corpus={}
vocub={}
vocub_main=[]
counter=1
for i in text_file:
    f=codecs.open(i,'r',encoding='utf_8',errors="ignore")
    print("Counter : ",counter)
    counter+=1
    c = f.readlines()
    final=[]
    for k in range(0,len(c)):

        final.append(c[k])
    #Final is a list of sentences 
#     tokenizer=RegexpTokenizer('\s+',gaps=True)
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    temp1=[]
    vocub={}
    for k in range(len(final)):
        temp=[]
        temp=tokenizer.tokenize(final[k])
        temp = copy.deepcopy([w.lower() for w in temp])
        temp=copy.deepcopy([lemmatizer.lemmatize(w,pos='v') for w in temp])
        temp=copy.deepcopy(list(set(temp)-set(stop_words)))
        
#             print("Stop Wro",temp)

        for n in range(len(temp)):


            t3=str(punc_improve(str(temp[n])))

            temp[n]=t3
            t5=''
            if temp[n].isdigit():
                t5=num2words(int(temp[n]))
            t6=t5.split(' ')
#             print("T6 : ",t6)
            t7=''
            if t6==[""] or t6==['']:
                pass
            else:
                for m in range(len(t6)):
                    t3=str(punc_improve(str(t6[m])))
                    t6[m]=t3
                    t1=str(dash_improve(str(t6[m])))
                    t6[m]=t1
                    t7=t7+t6[m]
                if temp[n]!='':

                    temp[n]=t7
            t1=str(dash_improve(str(temp[n])))
            temp[n]=t1
            if temp==list(['']) or temp==list([' ']) or temp[n]=='' or temp[n]==' ':
                pass
            else:
                temp1=temp1+temp
            if temp[n] in vocub.keys():
                vocub[temp[n]]=vocub[temp[n]]+1
            else:
                vocub[temp[n]]=1
    corpus[i]=vocub
    vocub_main=set(vocub_main).union(set(temp1))
    


# In[153]:


start4=time.time()
print(start4-start3)


# In[154]:


vocub_main=list(vocub_main)


# In[ ]:





# In[ ]:





# In[156]:


start6=time.time()


# In[157]:


idf_vector={}
idf=[]
tf_idf_main={}
counter=0
for j in vocub_main:

    count=0
    for i in corpus.keys():
        if counter%1000000==0:
            print("Counter : ",counter)
        counter+=1
        if j in corpus[i]:
            count+=1
        
    idf_vector[j]=count


# In[158]:



tdf_idf_main={}
counter=0
for i in corpus.keys():
    count=0
    f={}
    for j in vocub_main:
        if counter%1000000==0:
            print("Counter : ",counter)
        counter+=1
        f[j]=0
    tdf_idf_main[i]=f


# In[ ]:





# In[159]:


counter=0
for j in range(len(text_file)):
    count2=0
    counter+=1
    temp=[]
    print("Counter ",counter)
    tf_idf={}
    for i in range(len(vocub_main)):
        w=0.2
        if vocub_main[i] in title[j]:
            w=0.8
        temp=corpus[text_file[j]]
        if vocub_main[i] not in temp.keys():
            count1=0
        else:
            count1=temp[vocub_main[i]]
        tf1=count1/float(len(corpus[text_file[j]]))
        idf1=np.log(len(corpus)/(1+idf_vector[vocub_main[i]]))
        tf_f=tf1*idf1*w
        tdf_idf_main[text_file[j]][vocub_main[i]]=tf_f
        


# In[160]:


start7=time.time()
print(start7-start6)


# In[175]:


start8=time.time()


# In[205]:


query=input("Enter the Query : ")


# In[ ]:





# In[206]:


tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
temp1=[]
vocub={}
temp=[]
temp=tokenizer.tokenize(query)
temp = copy.deepcopy([w.lower() for w in temp])
temp=copy.deepcopy([lemmatizer.lemmatize(w,pos='v') for w in temp])
temp=copy.deepcopy(list(set(temp)-set(stop_words)))

#             print("Stop Words",temp)

for n in range(len(temp)):


    t3=str(punc_improve(str(temp[n])))

    temp[n]=t3
    t5=''
    if temp[n].isdigit():
        t5=num2words(int(temp[n]))
    t6=t5.split(' ')
#             print("T6 : ",t6)
    t7=''
    if t6==[""] or t6==['']:
        pass
    else:
        for m in range(len(t6)):
            t3=str(punc_improve(str(t6[m])))
            t6[m]=t3
            t1=str(dash_improve(str(t6[m])))
            t6[m]=t1
            t7=t7+t6[m]
        if temp[n]!='':

            temp[n]=t7
    t1=str(dash_improve(str(temp[n])))
    temp[n]=t1
    if temp==list(['']) or temp==list([' ']) or temp[n]=='' or temp[n]==' ':
        pass
    else:
        temp1.append(temp[n])
      


# In[ ]:





# In[207]:


relevant=[]
files=[]
for i in corpus.keys():
    s=0
    for j in temp1:
        try:
            s=s+tdf_idf_main[i][j]
        except:
            s=s+0
    relevant.append(s)
    files.append(i)
files_main=copy.deepcopy(files)
main1=sort_list(files,relevant)
main1.reverse()


# In[208]:


choice=int(input("Enter the number of relevant documents you want : "))


# In[209]:


print("Relevant Documents are : ",main1[:choice])


# In[210]:


temp2=copy.deepcopy(temp1)
temp1=set(temp1)
temp1=list(temp1)


# In[211]:


query_vector=[]

for i in range(len(vocub_main)):
    idf1=np.log(len(corpus)/float(1+idf_vector[vocub_main[i]]))
    count1=0
    if vocub_main[i] not in temp1:
        count1=0
    else:
        count1=temp2.count(vocub_main[i])
    tf1=count1/float(len(temp2))
    tf_f=tf1*idf1
    query_vector.append(tf_f)


# In[ ]:





# In[ ]:





# In[212]:


cosine_similarity=[]
q1=magnitude(query_vector)
for i in corpus.keys():
    temp=[]
    for j in vocub_main:
        temp.append(tdf_idf_main[i][j])
    cosine_similarity_val=np.dot(query_vector,temp)/float(q1*magnitude(temp))
    cosine_similarity.append(cosine_similarity_val)

    
    
        
        
    


# In[213]:


main1=sort_list(files_main,cosine_similarity)


# In[214]:


main1.reverse()


# In[215]:


print("Relevant Documents in reference with query are : ")
print(main1[:choice])


# In[216]:


start9=time.time()
print(start9-start8)


# In[ ]:





# In[ ]:





# In[ ]:




