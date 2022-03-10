from flask import Flask, request, make_response, render_template
import json
from flask_cors import cross_origin
# from logger import logger
import requests
from bs4 import BeautifulSoup
import re
from googlesearch import search
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import Counter
import operator
import math
from sklearn.linear_model import LogisticRegression
warnings.simplefilter("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download('all')

# df = pd.read_csv(io.BytesIO(uploaded['dis_sym_dataset_comb.csv']))
df = pd.read_csv("dis_sym_dataset_comb.csv")
# df1 = pd.read_csv(io.BytesIO(uploaded1['dis_sym_dataset_norm.csv']))
df1 = pd.read_csv("dis_sym_dataset_norm.csv")

X = df1.iloc[:, 1:]
Y = df1.iloc[:, 0:1]

# List of symptoms
dataset_symptoms = list(X.columns)
print(dataset_symptoms)
global final_symptoms
final_symptoms=[]

global final_symp
final_symp=[]

global final_symp2
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

    print("Inside webhook!!!")

    req = request.get_json(silent=True, force=True)
    print("Line 57 - ", req)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    print("Line 60 res - ", res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# processing the request from dialogflow
def processRequest(req):
    # log = logger.Log()

    print("Inside process request!!!")

    sessionID=req.get('responseId')

    result = req.get("queryResult")
    user_says=result.get("queryText")
    # log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print(result.get("outputContexts")[0])
    print(result.get("outputContexts")[0].get("parameters"))
    parameter2 = result.get("outputContexts")[0].get("parameters").get("symptoms")
    print(parameter2)
    print("Line 78 - ", parameters)

    intent = result.get("intent").get('displayName')

    def diseaseDetail(term):
        print("Inside diseaseDetail")
        diseases=[term]
        print("Line 85 and 86")
        print(diseases)
        ret=term+"\n"
        for dis in diseases:
                # search "disease wilipedia" on google
            query = dis+' wikipedia'
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


    if(intent=='symptoms-start'):
        # processed_user_symptoms=parameters.get("symptoms.original")
        processed_user_symptoms=parameters.get("symptoms")

        print("Line 127", processed_user_symptoms)

        fulfillmentText=""
        found_symptoms = set()
        # utlities for pre-processing
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        splitter = RegexpTokenizer(r'\w+')

        def synonyms(term):
            print("Inside synonyms")
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
            print(set(synonyms))
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
        print("After query expansion done by using the symptoms entered")
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
                    found_symptoms.add(data_sym)
        found_symptoms = list(found_symptoms)
        print(found_symptoms)
        STRfound_symptoms = ' '.join(map(str, found_symptoms))
        fulfillmentText= "This is the list of synonyms of your symptoms "+STRfound_symptoms+"  Enter indices."
        # for idx, symp in enumerate(found_symptoms):
        #     return(idx,":",symp)
        return {
            "fulfillmentText": fulfillmentText
        }


    if(intent=='symptoms-start-synonyms'):
        print("inside part 2")
        fulfillmentText=""

        term2= parameters.get("number")
        print(term2)

        # Show the related symptoms found in the dataset and ask user to select among them
        select_list = term2
        # Find other relevant symptoms from the dataset based on user symptoms based on the highest co-occurance with the
        # ones that is input by the user
        dis_list = set()
        # final_symp = []
        counter_list = []
        found_symptoms = list(parameter2)
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
        print(dict_symp_tup)

        # Iteratively, suggest top co-occuring symptoms to the user and ask to select the ones applicable
        found_symptoms=[]
     #   final_symptoms=[]
        count=0
        for tup in dict_symp_tup:
            count+=1
            found_symptoms.append(tup[0])
        final_symptoms.append(found_symptoms[0:10:])
        print(final_symptoms)
        STRfinal_symptoms = ' '.join(map(str, final_symptoms))

        fulfillmentText="This is a list of co-occuring symptoms"+STRfinal_symptoms+"Do you want to continue?"
        return {
            "fulfillmentText": fulfillmentText
            
        }

    if(intent=='symptoms-start-co-occuring'):

        term3=parameters.get("number")
        print(term3)

        fulfillmentText=""

        finals=final_symptoms
        # final_symp2=[]
        # terms=term3.split()
        print(finals)
        terms = term3
        print(terms)
        for i in range(len(terms)):
            x=int(terms[i])
            print(x)
            final_symp2.append(finals[0][x])
            
        print(final_symp2)
        for i in range(len(final_symp)):
            final_symp2.append(final_symp[i])

        STRfinal_symp2 = ' '.join(map(str, final_symp2))
        fulfillmentText="This is the final list of symptoms"+STRfinal_symp2+"Would you like to proceed?"
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
        # print(lr_pred)



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

        print('Line 361')
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
        fulfillmentText=diseaseDetail(my_array[my_arr.index(max)])
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
        print(fulfillmentText)



        # log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": fulfillmentText
        }

    else:
        # log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)
        print("else part")


if __name__ == '__main__':
    app.run(debug=True)
