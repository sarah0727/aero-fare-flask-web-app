from flask import Flask, render_template, request
import pickle
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model/aeroFare.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Dictionary mappings (from your training data)
airlines_mapping = {
    'Air India': 0,
    'GoAir': 1,
    'IndiGo': 2,
    'Jet Airways': 3,
    'Multiple carriers': 4,
    'SpiceJet': 5,
    'Vistara': 6,
    'Air Asia': 7,
    'Jet Airways Business': 8
}

city_mapping = {
    'Delhi': 0,
    'Kolkata': 1,
    'Mumbai': 2,
    'Chennai': 3,
    'Bangalore': 4
}

# Source cities (based on your one-hot encoding in the notebook)
source_cities = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']

def preprocess_input(form_data):
    # Extract date components
    journey_date = datetime.strptime(form_data['date'], '%Y-%m-%d')
    journey_day = journey_date.day
    journey_month = journey_date.month
    
    # Convert departure time to hours and minutes
    dep_time = datetime.strptime(form_data['dep_time'], '%H:%M')
    dep_hour = dep_time.hour
    dep_min = dep_time.minute
    
    # Convert arrival time to hours and minutes
    arrival_time = datetime.strptime(form_data['arrival_time'], '%H:%M')
    arrival_hour = arrival_time.hour
    arrival_min = arrival_time.minute
    
    # Calculate duration
    duration_hour = int(form_data['duration_hours'])
    duration_min = int(form_data['duration_minutes'])
    
    # Map categorical variables
    airline = airlines_mapping[form_data['airline']]
    destination = city_mapping[form_data['destination']]
    total_stops = int(form_data['stops'])
    
    # One-hot encode source
    source = form_data['source']
    source_encoded = [1 if city == source else 0 for city in source_cities]
    
    # Create feature array matching the model's expected input (16 features)
    features = np.array([
        airline,                    # Airline
        destination,               # Destination
        total_stops,              # Total_Stops
        journey_day,              # Journey_Day
        journey_month,            # Journey_Month
        dep_hour,                 # Dep_Time_Hrs
        dep_min,                  # Dep_Time_Min
        arrival_hour,             # Arrival_Time_Hrs
        arrival_min,              # Arrival_Time_Min
        duration_hour,            # Duration_Hrs
        duration_min             # Duration_Min
        ] + source_encoded        # Source city one-hot encoded (5 features)
    ).reshape(1, -1)
    
    return features

@app.route('/')
def home():
    return render_template('index.html',
                         airlines=list(airlines_mapping.keys()),
                         cities=list(city_mapping.keys()),
                         source_cities=source_cities)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        form_data = request.form
        
        # Preprocess input
        features = preprocess_input(form_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Round prediction to 2 decimal places
        prediction = round(prediction, 2)
        
        return render_template('result.html',
                             prediction=prediction,
                             form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)