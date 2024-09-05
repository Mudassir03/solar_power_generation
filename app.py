from flask import Flask, request, render_template
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Sample CSV data path
csv_file_path = 'data/solarpowergeneration.csv'

# Load and prepare the dataset
def load_data():
    df = pd.read_csv(csv_file_path)
    df = df.dropna()
    X = df.drop(['power_generated'], axis=1)
    y = df['power_generated']
    return X, y

# Train the model and return the trained model
def train_model(X, y):
    # Train the model
    gbr = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=200)
    gbr.fit(X, y)
    return gbr

# Load and prepare data
X, y = load_data()
gbr_model = train_model(X, y)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/estimation', methods=['POST'])
def estimation():
    try:
        # Get input data from the form
        distance_to_solar_noon = float(request.form['distance-to-solar-noon'])
        temperature = float(request.form['temperature'])
        wind_direction = float(request.form['wind-direction'])
        wind_speed = float(request.form['wind-speed'])
        sky_cover = float(request.form['sky-cover'])
        visibility = float(request.form['visibility'])
        humidity = float(request.form['humidity'])
        average_wind_speed = float(request.form['average-wind-speed-(period)'])
        average_pressure = float(request.form['average-pressure-(period)'])

        # Prepare the input data for prediction
        input_data = np.array([[distance_to_solar_noon, temperature, wind_direction, wind_speed,
                                sky_cover, visibility, humidity, average_wind_speed, average_pressure]])

        # Make the prediction
        prediction = gbr_model.predict(input_data)
        
        # Format the result
        if prediction[0] > 0:
            rslt = str(prediction[0].round(3)) + " J"
        else:
            rslt = 'No solar power generation is predicted for the given conditions.'
        
    except Exception as e:
        rslt = f"An error occurred: {str(e)}"

    # Return the prediction result
    return render_template('estimation.html', prediction=rslt)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
