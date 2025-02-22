from flask import Flask, render_template, request
from src.predict_pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    median_income = float(request.form.get('median_income'))
    housing_median_age = float(request.form.get('house_age'))
    total_rooms = float(request.form.get('total_rooms'))
    total_bedrooms = float(request.form.get('total_bedrooms'))
    population = float(request.form.get('population'))
    households = float(request.form.get('households'))
    latitude = float(request.form.get('latitude'))
    longitude = float(request.form.get('longitude'))
    ocean_proximity = request.form.get('ocean_proximity')
    
    # Prepare input data for prediction
    input_data = [[longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity]]
    
    # Call prediction pipeline
    predictor = PredictPipeline()
    predicted_price = predictor.predict(input_data)

    return render_template('home.html', results=f"${predicted_price}")

if __name__ == '__main__':
    app.run(debug=True)
