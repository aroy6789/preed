#import libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

#Initialize the flask App
app = Flask(__name__)
model_Total_Count = pickle.load(open('model_Total_Count.pkl', 'rb'))
model_Humidity = pickle.load(open('model_Humidity.pkl', 'rb'))
model_windspeed = pickle.load(open('model_windspeed.pkl', 'rb'))

#default page of our web-app
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
    
#To use the predict button in our web-app
standard_to = StandardScaler()
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction_1 = model_Total_Count.predict(final_features)
    prediction_2 = model_Humidity.predict(final_features)
    prediction_3 = model_windspeed.predict(final_features)

    output_1 = round(prediction_1[0], 2)
    output_2 = round(prediction_2[0], 2)
    output_3 = round(prediction_3[0], 2)
    
    return render_template('index.html',prediction_text="Predicted Rented-Bike Count : {}".format(output_1),prediction_text_1="Predicted Humidity : {}".format(output_2),prediction_text_2="Predicted Windspeed : {}".format(output_3))

if __name__ == "__main__":
    app.run(debug=True)
