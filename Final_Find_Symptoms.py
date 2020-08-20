#!/usr/bin/env python
# coding: utf-8

# In[100]:


import re
import numpy as np
import pandas as pd
from stemmer import Stemmer

data = pd.read_excel ("Gujarati_Dimensionality_Reduction.xlsx")

input_text =  input() 

s1="રોગ"
if input_text.startswith(s1): 
    flag=1
    stemmer = Stemmer()
    stemmed_text = stemmer.stem(input_text)
    stemmed_words = re.split(r'[;|,|\s]\s*', stemmed_text) 

    diseases_dire=[]

    for i in range(len(stemmed_words)): 
        for col in data['રોગ']: 
            if(stemmed_words[i]==col and stemmed_words[i]!='રોગ'):
                diseases_dire.append(stemmed_words[i])

    print("રોગો મેળ ખાતા ફોર્મ ડેટાસેટ : ",diseases_dire)
    diseases_dire=np.array(diseases_dire)

    data2 = pd.read_excel ("Gujarati_Dataset2.xlsx")

    Y = data2[data2.columns[0]].as_matrix()
    disease = Y.tolist()

    lendisease2=len(disease)

    columnnum=[]
    diseases_dire=np.array(diseases_dire)
    for i in range(lendisease2):
        for j in diseases_dire:
            if j ==  disease[i]:
                columnnum.append(i)

    disease_row=data2.loc[columnnum,:]
    disease_row=disease_row.to_numpy()
    #for i in range(len(columnnum)):
        
    #    print("\nરોગ : \n")
    #    print(disease_row[i][0])
    #    print("\nવર્ણન : \n")
    #    print(disease_row[i][1])
    #    print("\nલક્ષણો : \n")
    #    print(disease_row[i][2])
    #    print("\nઘરેલું ઉપાય : \n")
    #    print(disease_row[i][3])
    #    print("\nસારવાર : \n")
    #    print(disease_row[i][4])

else:       
    flag=0
    stemmer = Stemmer()
    input_text=input_text.replace("અને", ",")
    stemmed_text = stemmer.stem(input_text)
    stemmed_words = re.split(r'(;|,|\s)\s*', stemmed_text) 
    print()
    #print("સ્ટેમ પછી : ",stemmed_text)
   
    f = open("guj_pos_tag.txt",mode='r',encoding='UTF-8')
    data = f.read()
    sentences = data.split('\n')[1:]
    words = []
    for s in sentences:
        words.append(s.split('\t')[1])

    import re
    for i in range(len(sentences)):
        words[i] = re.sub(r'[^.A-Zઁ-૱\\,_-]',' ',words[i])
    pairs = []
    for i in range(len(words)):
        pairs.append(words[i].split(" "))
    tagged_guj_sentences = []

    for i in range(len(pairs)):
        for j in range(len(pairs[i])):
            if len(pairs[i][j].split("\\")) ==2:
                k,v = pairs[i][j].split("\\")
                tagged_guj_sentences.append((k,v))

    vocab=[word for word,tag in tagged_guj_sentences]
    tags=[tag for word,tag in tagged_guj_sentences]

    from stemmer import Stemmer
    stemmer = Stemmer()
    stem_words = []
    for v in vocab:
        stem_words.append(stemmer.stem(v))
    noun_words_intxt = dict(zip(stem_words,tags))
    noun_words_intxt = [(k, v) for k, v in noun_words_intxt.items()]  #convert to list
    #print("પોસ્ટગર : ",noun_words_intxt)

    from difflib import SequenceMatcher
    compared_words=[]
    maxn=0
    for j in range(len(stemmed_words)):
        for i in range(len(noun_words_intxt)):
            if(noun_words_intxt[i][0]!='મ'):
                if(noun_words_intxt[i][1]=='N_NN' or noun_words_intxt[i][1]=='RD_PUNC' ):
                    #print(noun_words_intxt[i][0],stemmed_words[j])
                    if(stemmed_words[j]==noun_words_intxt[i][0]):
                        #print(noun_words_intxt[i][0],stemmed_words[j])
                        compared_words.append(stemmed_words[j])
    #print()
    #print("ફક્ત નોઉન્સ : ",compared_words)
    #print()

    def listToString(s):  

        str1 = ""  

        for ele in s:  
            str1 =str1+ele+" "   

        return str1  

    listnoun=listToString(compared_words)
    symptoms_byuser =  re.split(r'[,]\s*', listnoun) 
    symptoms_byuser=np.array(symptoms_byuser)
    for i in range(len(symptoms_byuser)):
        symptoms_byuser[i]=symptoms_byuser[i].strip()
        
    symptoms_byuser=symptoms_byuser.tolist()
    #print("વપરાશકર્તાઓ દ્વારા લક્ષણો : ",symptoms_byuser)
    #print()

    symptoms_byuser=np.array(symptoms_byuser)

    from difflib import SequenceMatcher
    
    data = pd.read_excel ("Gujarati_Dimensionality_Reduction.xlsx")
    symptoms=[]
    max_n=[]
    symptom=0

    for i in range(len(symptoms_byuser)): 
        maxn=0
        for col in data.columns: 
            #print(symptoms_byuser[i])
            ratio = SequenceMatcher(None, col, symptoms_byuser[i]).ratio()
            if(ratio!=0 and maxn<ratio):
                maxn=ratio
                symptom=col
        max_n.append(maxn*100)
        symptoms.append(symptom)
        print(symptom)
        print(maxn*100,"%")

    #print("લક્ષણો ડેટાસેટ સાથે મેળ ખાતા : ",symptoms,max_n)
    #print("લક્ષણો અને એની સંભાવનાઓ : ",symptoms,max_n)
      
        
    #મને પેટ નો દુખાવો થાય છે ,મને માથુ પણ દુખે છે 
    #મને ત્વચા પર ચકામા, થાક ,ખંજવાળ અને વધારે તાવ આવે છે
    #રોગ ડાયાબિટીસ અને ટાઇફોઇડ થયો છે
    #મને ગળામાં ખંજવાળ આઈ રહી છે અને ગળામાં પેચ જેવુ પણ લાગે છે


# In[101]:


if flag == 0:

    # making data frame
    data = pd.read_excel ("Gujarati_Dimensionality_Reduction.xlsx")
    i=0
    a = []

    # iterating the columns (symptoms-133)
    for col in data.columns:
        a.append(col) 
        i=i+1

    a.pop() # Detele last column which is for diseases
    l=len(a)
    s = [0] * l # Generate empty array s of size 132

    #symptoms=input('લક્ષણો દાખલ કરો  : ')

    #print(type(s))
    import nltk
    symptoms_word=symptoms
    for i in symptoms_word:
        s[a.index(i)]=1

    s=np.array(s)
    s=s.reshape(-1,1)
    s.shape
    s=s.T

    alldiseases = []
    for i in symptoms_word: 
        for k in range (41):
            if data[i][k] == 1:    
                alldiseases.append(data['રોગ'][k])  # All the disease of entered input

    def countfreq(alldiseases):
        freq_diseases = dict()

        for elem in alldiseases:
            # If element exists in dict then increment its value 
            if elem in freq_diseases:
                freq_diseases[elem] += 1
            else:
                freq_diseases[elem] = 1    

        freq_diseases = { key:value for key, value in freq_diseases.items() if value >= len(symptoms_word)}
        # Returns a dict of duplicate elements and thier frequency count
        return freq_diseases

    freq_diseases = countfreq(alldiseases)   
    freq_diseases = list(freq_diseases.items())
    commondiseases = np.array(freq_diseases)

    data = pd.DataFrame(commondiseases)
    data=data[data.columns[:-1]]
    commondiseases=data.to_numpy()

    print()
    if len(commondiseases) != 0:
        print("તમને થઈ શકે તેવા રોગની સંભાવનાઓ : ")
        alldiseases=commondiseases
        print(alldiseases)   
    else:
        x = np.array(alldiseases) 
        alldiseases=np.unique(x)
        print("તમને થઈ શકે તેવા રોગની સંભાવનાઓ : ")
        print(alldiseases)

    #ખાંસી કફ થાક 
    #ત્વચા_પર_ચકામા થાક ખંજવાળ સુસ્તી વધારે_તાવ


# In[107]:


if flag == 0:
    data = pd.read_excel ("Gujarati_Dimensionality_Reduction.xlsx")
    disease=data.iloc[:, -1]
    len_disease=len(disease)

    len_alldisease=len(alldiseases)
    Number=[];

    j=0
    for i in range(len_disease):
        if j != len_alldisease:
            if alldiseases[j] ==  disease[i]:
                Number.append(i)
                j=j+1
        i=i+1

    feature_row=data.loc[Number,:]
    feature_row=feature_row.to_numpy()

    features=[]
    for j in range(len_alldisease):
        if j !=len_alldisease:
            for i in range(128):
                if feature_row[j][i] == 1:
                    features.append(a[i])
                i=i+1

    for k in range(len(symptoms_word)):
        j=0
        for j in features:
            if symptoms_word[k] == j:
                features.remove(j);           
        k=k+1   

    features=np.unique(features)

    if features != []:
        print("આ અન્ય લક્ષણો છે : ")
        print(features)
        print()
        print("તમારાથી સંબંધિત અન્ય કોઈ લક્ષણો કૃપા કરીને દાખલ કરો હા-1 , નાં-0, અટકાવવા માટે-stop")
    print()

    #s=s.T
    #user_features=input('લક્ષણો દાખલ કરો  : ')
    #import nltk
    #usersymptoms=nltk.word_tokenize(user_features)

    #for i in usersymptoms:
    #    s[a.index(i)]=1

    #print(s,s.shape,type(s))

    lst = [ ] 
    n = len(features)
    s=s.T

    for m in features: 
        print(m)
        ele = str(input())
        if ele == "stop":
            break
        if ele != "0":
            s[a.index(m)]=1
        if ele!="1" and ele !="0":
            break


    # હા-0 , નાં-1  #મને શ્વાસ ચડી રહ્યો છે , મને સાઇનસ પર દબાણ થઈ રહ્યું છે
    # અસ્વસ્થતા કાટવાળું_ગળફામ ઠંડી વધારે_તાવ વાસ પરસેવો  છાતીનો_દુખાવો ઝડપી_ધબકારા ાં સતત_છીંક


# In[108]:


if flag == 0:
    s=s.T
    #print(s,s.shape,type(s))

    #print(feature_row,s)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    data_temp = pd.read_excel ("Gujarati_Training.xlsx")
    X = data_temp.iloc[:, :-1].values
    Y = data_temp.iloc[:, -1].values
    #print(X,Y)
    y = labelencoder.fit_transform(Y)

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X, y)
    pr1= classifier.predict(s)
    R1=labelencoder.inverse_transform(pr1)
    
    data = pd.read_excel ("Gujarati_Dimensionality_Reduction.xlsx")
    disease_original=data.loc[data['રોગ'].isin(R1)]
    temp=disease_original.drop(columns=['રોગ'])
    disease_original = np.array(temp)
    s=np.array(s)

    count=0
    total=0

    for j in range(len(disease_original.T)):
        if (disease_original.T[j]==1):
            total=total+1

    #print(total)
    for i in range(len(disease_original.T)):
        if (s.T[i]==disease_original.T[i] and disease_original.T[i]==1):
            count=count+1

    prp=(count/total)*100;
    print("દાખલ કરેલા લક્ષણો પર થી ",R1," થવાની સંભાવના છે : ",prp,"%");
    


# In[109]:


if flag == 0:
    import pandas as pd

    data2 = pd.read_excel ("Gujarati_Dataset2.xlsx")

    Y = data2[data2.columns[0]].as_matrix()
    disease = Y.tolist()

    lendisease2=len(disease)

    columnnum=[]
    j=0
    for i in range(lendisease2):
        if R1 ==  disease[i]:
            columnnum.append(i)

    disease_row=data2.loc[columnnum,:]
    disease_row=disease_row.to_numpy()
    print(disease_row[0][0])
    print("\nશું તમે વર્ણન વિશે જાણવા માંગો છો ? (હા / નાં) : \n")
    Description = str(input())
    if Description != '0':
        print(disease_row[0][1])

    print("\nશું તમે લક્ષણો વિશે જાણવા માંગો છો ? (હા / નાં) : \n")
    Description = str(input())
    if Description != '0':
        print(disease_row[0][2])

    print("\nશું તમે ઘરેલું ઉપાય વિશે જાણવા માંગો છો ? (હા / નાં) : \n")
    Description = str(input())
    if Description != '0':
        print(disease_row[0][3])

    print("\nશું તમે સારવાર વિશે જાણવા માંગો છો ? (હા / નાં) : \n")
    Description = str(input())
    if Description != '0':
        print(disease_row[0][4])

    # હા-1 , નાં-0

