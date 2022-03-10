# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, make_response, render_template
from flask_cors import cross_origin
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
# from statistics import mean
# from collections import Counter
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

# df = pd.read_csv("dis_sym_dataset_norm.csv")
# df1 = pd.read_csv("dis_sym_dataset_comb.csv")

# X = df1.iloc[:, 1:]
# Y = df1.iloc[:, 0:1]

# List of symptoms
# dataset_symptoms = list(X.columns)
# print(dataset_symptoms)

# Global Lists
# global final_symptoms
# global final_symp
# global final_symp2
# final_symptoms=[]
# final_symp=[]
# final_symp2=[]

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
    parameters = result.get("parameters").get("symptoms")
#     parameter2 = result.get("outputContexts")[0].get("parameters").get("symptoms")
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
        final_symp2=parameters


       
       
        fulfillmentText = "You may have one of these following diseases: \n\n"
        # fulfillmentText = my_array[my_arr.index(max)]
        for i in range(len(final_symp2)):
            print(final_symp2)   
            fulfillmentText += diseaseDetail(final_symp2[i])
        fulfillmentText += "We suggest consulting a real doctor before starting any treatment for your own safety!"
        return {
            "fulfillmentText": fulfillmentText
        }


    elif(intent=='FAQ - disease-info'):
        disease_info = parameters
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
