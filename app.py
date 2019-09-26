from flask import Flask, abort, jsonify, render_template,url_for, request,send_from_directory,redirect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np 
import pandas as pd 
import json
import requests 

dfDigi = pd.read_json('digimon.json')
# Rekomendasi 4 digimon based on stage, type & attribute
def digimon(i):
    return str(i['stage']) + 'shendy' + str(i['type'])+ 'shendy' + str(i['attribute'])
dfDigi['data_digi'] = dfDigi.apply(digimon, axis=1)
dfDigi['digimon'] = dfDigi['digimon'].apply(lambda i:i.lower())

model = CountVectorizer(tokenizer = lambda dfDigi: dfDigi.split('shendy'))
matrix = model.fit_transform(dfDigi['data_digi'])
score = cosine_similarity(matrix)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST','GET'])
def result():
    body = request.form
    digimon = body['digimon']
    digimon = digimon.lower()

    if digimon not in list(dfDigi['digimon']):
        return redirect('/notfound')
    
    indexDigi = dfDigi[dfDigi['digimon'] == digimon].index.values[0]
    samasuka = dfDigi.iloc[indexDigi][['Nama','stage','type','attribute','image']]
    scoreDigi = list(enumerate(score[indexDigi]))
    sortDigi = sorted(scoreDigi, key = lambda i:i[1], reverse = True)
    rekomendasi =[]
    for item in sortDigi[:7]:
        digi_x = {}
        if dfDigi.iloc[item[0]['digimon'] != digimon]:
            name = dfDigi.iloc[item[0]]['digimon'].capitalize()
            stage = dfDigi.iloc[item[0]]['stage']
            image = dfDigi.iloc[item[0]]['image']
            tipe = dfDigi.iloc[item[0]]['type']
            attribute = dfDigi.iloc[item[0]['attribute']]
            digi_x['digimon'] = name
            digi_x['stage'] = stage
            digi_x['image'] = image
            digi_x['type'] = tipe
            digi_x['attribute'] = attribute
            rekomendasi.append(digi_x)
    return render_template('result.html',rekomendasi = rekomendasi, samasuka = samasuka)


@app.route('/notfound')
def notFound():
    return render_template('error.html')


# @app.errorhandler(404)
# def page_not_found(error):
# 	return render_template('error.html')

if __name__ == "__main__":
    # model = joblib.load('')
    app.run(debug=True)