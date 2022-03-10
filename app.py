# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, make_response, render_template
from flask_cors import cross_origin
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from statistics import mean
from collections import Counter
from bs4 import BeautifulSoup
from googlesearch import search
from time import time
import operator
import math
import requests
import re
import numpy as np
import pandas as pd
import json
import nltk
# nltk.download('all')
# nltk.download('wordnet', quiet=True)
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# df = pd.read_csv("encodedDataset.csv")
# df1 = pd.read_csv("encodedDataset-2.csv")

df = pd.read_csv("dis_sym_dataset_norm.csv")
df1 = pd.read_csv("dis_sym_dataset_comb.csv")

X = df1.iloc[:, 1:]
Y = df1.iloc[:, 0:1]

# List of symptoms
dataset_symptoms = list(X.columns)
# print(dataset_symptoms)

# Global Lists
global final_symptoms
global final_symp
global final_symp2
final_symptoms=[]
final_symp=[]
final_symp2=[]

# Flask Code
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():
    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    print("Line 62 response - ", res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

# processing the request from dialogflow
def processRequest(req):

    print("Line 74 - Processing request")
    sessionID=req.get('responseId')
    result = req.get("queryResult")
    parameters = result.get("parameters")
    parameter2 = result.get("outputContexts")[0].get("parameters").get("symptoms")
    print("Line 79 - ", parameters)
    intent = result.get("intent").get('displayName')
    print("Line 81 - Intent Name ", intent)

    def diseaseDetail(term):
        print("Line 84 - Finding details")
        diseases=[term]
        ret=term+"\n"
        for dis in diseases:
            query = dis+' wikipedia'
            for sr in search(query):
                match=re.search(r'wikipedia',sr)
                filled = 0
                if match:
                    wiki = requests.get(sr,verify=False)
                    soup = BeautifulSoup(wiki.content, 'html5lib')
                    # Fetch HTML code for 'infobox'
                    info_table = soup.find("table", {"class":"infobox"})
                    if info_table is not None:
                        for row in info_table.find_all("tr"):
                            data=row.find("th",{"scope":"row"})
                            if data is not None:
                                symptom=str(row.find("td"))
                                symptom = symptom.replace('.','')
                                symptom = symptom.replace(';',',')
                                symptom = symptom.replace('<b>','<b> \n')
                                symptom=re.sub(r'<a.*?>','',symptom) # Remove hyperlink
                                symptom=re.sub(r'</a>','',symptom) # Remove hyperlink
                                symptom=re.sub(r'<[^<]+?>',' ',symptom) # All the tags
                                symptom=re.sub(r'\[.*\]','',symptom) # Remove citation text
                                symptom=symptom.replace("&gt",">")
                                ret += data.get_text() + " - " + symptom +"\n\n"
                                filled = 1
                        if filled:
                            break
        return ret

    # First Time Webhook call
    if(intent=='symptoms-start'):
        processed_user_symptoms=parameters.get("symptoms")
        print("Line 119 - User symptoms are ", processed_user_symptoms)

        fulfillmentText=""
        found_symptoms = set()

        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        splitter = RegexpTokenizer(r'\w+')

        def synonyms(term):
            synonyms = []
            response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
            soup = BeautifulSoup(response.content,  "html.parser")
            try:
                container=soup.find('section', {'class': 'MainContentContainer'})
                row=container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
                row = row.find_all('li')
                for x in row:
                    synonyms.append(x.get_text())
            except:
                None
            for syn in wordnet.synsets(term):
                synonyms+=syn.lemma_names()
            return set(synonyms)

    # Taking each user symptom and finding all its synonyms and appending it to the pre-processed symptom string
        user_symptoms = []
        for user_sym in processed_user_symptoms:
            user_sym = user_sym.split()
            str_sym = set()
            for comb in range(1, len(user_sym)+1):
                for subset in combinations(user_sym, comb):
                    subset=' '.join(subset)
                    subset = synonyms(subset)
                    str_sym.update(subset)
            str_sym.add(' '.join(user_sym))
            user_symptoms.append(' '.join(str_sym).replace('_',' '))
        # query expansion performed by joining synonyms found for each symptoms initially entered
        print("Line 157 - After query expansion done by using the symptoms entered")
        print(user_symptoms)

        # Loop over all the symptoms in dataset and check its similarity score to the synonym string of the user-input
        # symptoms. If similarity>0.5, add the symptom to the final list
        for idx, data_sym in enumerate(dataset_symptoms):
            data_sym_split=data_sym.split()
            for user_sym in user_symptoms:
                count=0
                for symp in data_sym_split:
                    if symp in user_sym.split():
                        count+=1
                if count/len(data_sym_split)>0.5:
                    if data_sym not in found_symptoms:
                        found_symptoms.add(data_sym)
                    
        found_symptoms = list(found_symptoms)
        print("Line 174 - found_symptoms :")
        print(found_symptoms)

        def defSTRfound_symptoms():
            fulfillmentText = "\n"
            for idx, value in enumerate(found_symptoms):
                index = int(idx)+1
                tup =  (index, ":", value, "\n")
                STRfound_symptoms = ' '.join(map(str, tup))
                fulfillmentText += STRfound_symptoms
            print(fulfillmentText)     
            return fulfillmentText
        
        fulfillmentText= "This is the list of synonyms of your symptoms - "+ defSTRfound_symptoms() +" Enter the applicable indices. \n In the form - I have 1,2,3 "
        return {
            "fulfillmentText": fulfillmentText
        }


    if(intent=='symptoms-start-synonyms'):
        fulfillmentText=""
#         print("Line 195 - found_symptoms :")
#         print(found_symptoms)
        term2 = []
        userinput = parameters.get("number")
        print(userinput )
        for i in range(0, len(userinput)):
            term2.append(int(userinput[i])-1)
        print("Line 202 - User entered indices after processing - ", term2)
      
        # Show the related symptoms found in the dataset and ask user to select among them
        select_list = term2
        # Find other relevant symptoms from the dataset based on user symptoms based on the highest co-occurance with the
        # ones that is input by the user
        dis_list = set()
        # final_symp = []
        counter_list = []
        found_symptoms = list(parameter2)
        print("Line 212 - parameter2 - found_symptoms:")
        print(found_symptoms)
        for idx in select_list:
            symp=found_symptoms[int(idx)]
            final_symp.append(symp)
            dis_list.update(set(df1[df1[symp]==1]['label_dis']))

        for dis in dis_list:
            row = df1.loc[df1['label_dis'] == dis].values.tolist()
            row[0].pop(0)
            for idx,val in enumerate(row[0]):
                if val!=0 and dataset_symptoms[idx] not in final_symp:
                    counter_list.append(dataset_symptoms[idx])

        # Symptoms that co-occur with the ones selected by user
        dict_symp = dict(Counter(counter_list))
        dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1),reverse=True)
        # print(dict_symp_tup)         #Lists all the cooccuring symptoms

        # Iteratively, suggest top co-occuring symptoms to the user and ask to select the ones applicable
        found_symptoms=[]

        count=0
        for tup in dict_symp_tup:
            count+=1
            found_symptoms.append(tup[0])
        final_symptoms.append(found_symptoms[0:10:])
        print("Line 239 - ")
        print(final_symptoms)

        def defSTRfinal_symptoms():
            fulfillmentText = "\n"
            for idx, value in enumerate(final_symptoms[0]):
                index = int(idx)+1
                tup =  (index, ":", value, "\n")
                STRfinal_symptoms = ' '.join(map(str, tup))
                fulfillmentText += STRfinal_symptoms
                print(fulfillmentText)     
            return fulfillmentText

        fulfillmentText="This is a list of co-occuring symptoms - \n" + defSTRfinal_symptoms() + " Enter the applicable indices. \n In the form - Choose 1,2,3 "
        return {
            "fulfillmentText": fulfillmentText
            
        }

    if(intent=='symptoms-start-co-occuring'):

        term3 = []
        userinput = parameters.get("number")
        print(userinput )
        for i in range(0, len(userinput)):
            term3.append(int(userinput[i])-1)
        print("Line 265 - User entered indices after processing - ", term3)

        fulfillmentText=""
        print(final_symptoms)
        finals=final_symptoms[0]
        print("Line 270 - finals :", finals)
        for i in range(len(term3)):
            x=int(term3[i])
            final_symp2.append(finals[x])
            
        print(final_symp2)
        
        for i in range(len(final_symp)):
            final_symp2.append(final_symp[i])

        def defSTRfinal_symp2():
            fulfillmentText = "\n"
            for idx, value in enumerate(final_symp2):
                index = int(idx)+1
                tup =  (index, ":", value, "\n")
                STRfinal_symp2 = ' '.join(map(str, tup))
                fulfillmentText += STRfinal_symp2
                print(fulfillmentText)     
            return fulfillmentText

        fulfillmentText="This is the final list of symptoms - \n"+ defSTRfinal_symp2() + "Type YES to proceed. \n"
        return {
            "fulfillmentText": fulfillmentText
        }

    elif(intent=='symptoms-start-co-occuring - yes'):

        fulfillmentText=""

        sample_x = [0 for x in range(0,len(dataset_symptoms))]
        for val in final_symp2:
            print(val)
            sample_x[dataset_symptoms.index(val)]=1
        
        # Predict disease
        print(final_symp2)
        lr = LogisticRegression()
        lr = lr.fit(X, Y)
        lr_pred = lr.predict_proba([sample_x])

        # Predict disease
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=4)
        knn = knn.fit(X, Y)
        knn_pred = knn.predict_proba([sample_x])

        # rf = RandomForestClassifier(n_estimators=10, criterion='entropy')
        # rf = rf.fit(X, Y)
        # rf_pred = rf.predict_proba([sample_x])

        # Multinomial NB Classifier
        mnb = MultinomialNB()
        mnb = mnb.fit(X, Y)
        mnb_pred = mnb.predict_proba([sample_x])

        k = 1
        diseases = list(set(Y['label_dis']))
        diseases.sort()

        # topkrf = rf_pred[0].argsort()[-k:][::-1]
        topkmnb = mnb_pred[0].argsort()[-k:][::-1]
        topkknn = knn_pred[0].argsort()[-k:][::-1]
        topklr = lr_pred[0].argsort()[-k:][::-1]
        # print(topk)


        # Take input a disease and return the content of wikipedia's infobox for that specific disease

        # print(f"\nTop {k} diseases predicted based on symptoms")
        topk_dict = {}

        my_array = []
        my_arr = []

        # Show top 10 highly probable disease to the user.
        for idx,t in  enumerate(topkmnb):
            # print(idx, t)
            match_sym=set()
            row = df1.loc[df1['label_dis'] == diseases[t]].values.tolist()
            # print(row)

            my_array.append(row[0].pop(0))
            # row[0].pop(0)


            # PROBABILITY CALCULATE SOCHOOOOOOOOOOOOOOOOOOOOOOOOOOOO
            for idx,val in enumerate(row[0]):
                # print(idx, val)
                if val!=0:
                    match_sym.add(dataset_symptoms[idx])
            prob = (len(match_sym.intersection(set(final_symp2)))+1)/(len(set(final_symp2))+1)
            # prob *= mean(scores)
            topk_dict[t] = prob
            my_arr.append(prob)

        for idx,t in  enumerate(topkknn):
            # print(idx, t)
            match_sym=set()
            row = df1.loc[df1['label_dis'] == diseases[t]].values.tolist()

            my_array.append(row[0].pop(0))
            for idx,val in enumerate(row[0]):
                # print(idx, val)
                if val!=0:
                    match_sym.add(dataset_symptoms[idx])
            prob = (len(match_sym.intersection(set(final_symp2)))+1)/(len(set(final_symp2))+1)
            # prob *= mean(scores)
            # topk_dict[t] = prob
            my_arr.append(prob)



        for idx,t in  enumerate(topklr):
            # print(idx, t)
            match_sym=set()
            row = df1.loc[df1['label_dis'] == diseases[t]].values.tolist()
            # print(row)

            my_array.append(row[0].pop(0))
            for idx,val in enumerate(row[0]):
                # print(idx, val)
                if val!=0:
                    match_sym.add(dataset_symptoms[idx])
            prob = (len(match_sym.intersection(set(final_symp2)))+1)/(len(set(final_symp2))+1)
            # prob *= mean(scores)
            # topk_dict[t] = prob
            my_arr.append(prob)

        print('Line 389')
        print(my_arr)
        print(my_array)
        #Initialize max with first element of array.
        max = my_arr[0]

        #Loop through the array
        for i in range(0, len(my_arr)):
            #Compare elements of array with max
            if(my_arr[i] > max):
                max = my_arr[i]
        print('Line 372')
        print(my_arr.index(max))
        print()
        fulfillmentText = "You may have one of these following diseases: \n\n"
        fulfillmentText += diseaseDetail(my_array[my_arr.index(max)])
        fulfillmentText += "We suggest consulting a real doctor before starting any treatment for your own safety!"
        return {
            "fulfillmentText": fulfillmentText
        }


    elif(intent=='FAQ - disease-info'):
        disease_info = parameters.get("symptoms")
        fulfillmentText=""

        def diseaseDetail(term):
            print(term[0])
            ret="  \n"
            # for dis in diseases:
            # search "disease wilipedia" on google
            query = term[0] +' wikipedia'
            # tld="co.in"
            # ,stop=10,pause=0.5
            for sr in search(query):
                # open wikipedia link
                match=re.search(r'wikipedia',sr)
                filled = 0
                if match:
                    wiki = requests.get(sr,verify=False)
                    soup = BeautifulSoup(wiki.content, 'html5lib')
                    # Fetch HTML code for 'infobox'
                    info_table = soup.find("table", {"class":"infobox"})
                    if info_table is not None:
                        # Preprocess contents of infobox
                        for row in info_table.find_all("tr"):
                            data=row.find("th",{"scope":"row"})
                            if data is not None:
                                symptom=str(row.find("td"))
                                symptom = symptom.replace('.','')
                                symptom = symptom.replace(';',',')
                                symptom = symptom.replace('<b>','<b> \n')
                                symptom=re.sub(r'<a.*?>','',symptom) # Remove hyperlink
                                symptom=re.sub(r'</a>','',symptom) # Remove hyperlink
                                symptom=re.sub(r'<[^<]+?>',' ',symptom) # All the tags
                                symptom=re.sub(r'\[.*\]','',symptom) # Remove citation text
                                symptom=symptom.replace("&gt",">")
                                ret+=data.get_text()+" - "+symptom+"\n"
                                # print(data.get_text(),"-",symptom)
                                filled = 1
                    if filled:
                        break
            return ret


        detail =  "Requested information -  \n"+ diseaseDetail(disease_info)
        fulfillmentText += detail
        print("Line 464 - ", fulfillmentText)

        return {
            "fulfillmentText": fulfillmentText
        }

    else:
        print("else part")


if __name__ == '__main__':
    app.run(debug=True)
