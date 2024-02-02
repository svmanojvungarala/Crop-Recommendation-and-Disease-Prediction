# Importing Required Modules
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))

# Importing the trained picle model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()



crop_recommendation_model_path = 'rf_model.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Function to fetch latitude and longitude of given city

def fetch_coordinates(city_name,state_name,country_name):
    """
    Fetch and returns the latitude,longitude of a city
    :params: city_name,state_name,country_name
    :return: latitude,longitude
    """
    api_key = open('geo_api_key.txt','r').read()

    city = city_name
    state = state_name
    country = country_name    


    urlx = "http://api.openweathermap.org/geo/1.0/direct?q={city}+,{state},{country}&limit=1&appid={api_key}".format(city=city,state=state,country=country,api_key=api_key)
    geo_api_response = requests.get(urlx).json()
    latitude = geo_api_response[0]['lat']
    longitude =geo_api_response[0]['lon']
    return latitude,longitude

# Function to fetch weather given latitude and longitude

def fetch_weather(latitude,longitude):
    """
    Fetch and returns the temperature and humidity of cordinates
    params : latitude,longitude
    :return: temperature, humidity
    """
    
    api_key = open('weather_api_key.txt','r').read()

    lat =latitude
    lon = longitude

    urlx = "https://api.agromonitoring.com/agro/1.0/weather/forecast?lat={lat}&lon={lon}&appid={api_key}".format(api_key=api_key,lat=lat,lon=lon)
    weather_api_response = requests.get(urlx).json()

    humidity    = weather_api_response[1]['main']['humidity']
    temperature_kelvin = weather_api_response[1]['main']['temp']
    temperature_celsius = temperature_kelvin - 273.15

    return humidity,temperature_celsius


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

@app.route('/')
def home():
    return render_template('crop.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosphorous'])
        K = int(request.form['Pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        
        # Calling the fetch_coordinates and fetch_weather functions

        lat,lon = fetch_coordinates(city,state,country)
        humidity,temperature=fetch_weather(lat,lon)

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = model.predict(data)
        final_prediction = my_prediction[0]
        
        return render_template('crop.html', prediction_text='The Best suitable crop is {}'.format(final_prediction), title='Harvestify - Home')


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

if __name__ == "__main__":
    app.run(debug=True)
