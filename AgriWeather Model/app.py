from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import webbrowser
from threading import Timer

app = Flask(__name__)
CORS(app)

# ==========================================
# LOAD TRAINED MACHINE LEARNING MODEL
# ==========================================

model = joblib.load('crop_yield_model.joblib')

print("AgriWeather ML model loaded successfully!")


# ==========================================
# CROPS
# ==========================================

CROPS = [
    "Maize",
    "Wheat",
    "Soybean",
    "Sunflower",
    "Groundnuts",
    "Beans",
    "Cotton",
    "Sorghum"
]


# ==========================================
# HOME PAGE
# ==========================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

# ==========================================
# PREDICTION API
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get data from website
        data = request.get_json()

        rainfall = float(data.get('rainfall', 0))
        frost_level = float(data.get('frost', 0))
        temperature = float(data.get('temperature', 0))
        solar_radiation = float(data.get('solar', 0))
        wind_speed = float(data.get('wind', 0))


        # ==========================================
        # PREDICT YIELD FOR ALL CROPS
        # ==========================================

        predictions = []

        for crop in CROPS:

            scenario = pd.DataFrame([{
                'crop': crop,
                'rainfall': rainfall,
                'frost_level': frost_level,
                'temperature': temperature,
                'solar_radiation': solar_radiation,
                'wind_speed': wind_speed
            }])

            predicted_yield = model.predict(scenario)[0]

            # Prevent negative yields
            predicted_yield = max(0, predicted_yield)

            predictions.append({
                "crop": crop,
                "yield": round(float(predicted_yield), 2)
            })


        # ==========================================
        # SORT BY PREDICTED YIELD
        # ==========================================

        predictions.sort(
            key=lambda x: x["yield"],
            reverse=True
        )


        # ==========================================
        # BEST CROP
        # ==========================================

        recommended_crop = predictions[0]["crop"]
        recommended_yield = predictions[0]["yield"]


        # ==========================================
        # FEATURE IMPORTANCE
        # ==========================================

        # Get the XGBoost regressor
        xgb_model = model.named_steps['regressor']

        # Get preprocessor
        preprocessor = model.named_steps['preprocessor']

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()

        importance_values = xgb_model.feature_importances_

        feature_importance = []

        for name, importance in zip(
            feature_names,
            importance_values
        ):

            # Clean feature names
            clean_name = name.replace(
                "num__",
                ""
            ).replace(
                "cat__crop_",
                "Crop: "
            )

            feature_importance.append({
                "name": clean_name,
                "weight": round(
                    float(importance) * 100,
                    2
                )
            })


        # Sort feature importance
        feature_importance.sort(
            key=lambda x: x["weight"],
            reverse=True
        )


        # ==========================================
        # RETURN RESULTS
        # ==========================================

        return jsonify({

            "predictions": predictions,

            "recommended_crop": recommended_crop,

            "recommended_yield": recommended_yield,

            "feature_importance": feature_importance

        })


    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500

# ==========================================
# CROP YIELD HEAT MAP API
# ==========================================

@app.route('/heatmap', methods=['POST'])
def generate_heatmap():

    try:
        data = request.get_json()

        # Read the selected crop
        crop = data.get('crop', 'Maize')

        # Validate the selected crop
        if crop not in CROPS:
            return jsonify({
                "error": "Invalid crop selected."
            }), 400

        # Keep the other environmental inputs fixed
        frost = float(data.get('frost', 1))
        solar = float(data.get('solar', 21))
        wind = float(data.get('wind', 10))

        # Rainfall grid: 200 mm to 1400 mm
        rainfall_values = list(range(200, 1401, 100))

        # Temperature grid: 12°C to 38°C
        temperature_values = list(range(12, 39, 2))

        scenarios = []

        # Generate every rainfall-temperature combination
        for temperature in temperature_values:

            for rainfall in rainfall_values:

                scenarios.append({
                    'crop': crop,
                    'rainfall': rainfall,
                    'frost_level': frost,
                    'temperature': temperature,
                    'solar_radiation': solar,
                    'wind_speed': wind
                })

        # Use the existing trained model
        scenario_df = pd.DataFrame(scenarios)

        predictions = model.predict(scenario_df)

        # Construct the heat map matrix
        yield_matrix = []
        index = 0

        for temperature in temperature_values:

            row = []

            for rainfall in rainfall_values:

                predicted_yield = max(
                    0,
                    float(predictions[index])
                )

                row.append(round(predicted_yield, 2))

                index += 1

            yield_matrix.append(row)

        # Return results to the website
        return jsonify({
            "crop": crop,
            "rainfall_values": rainfall_values,
            "temperature_values": temperature_values,
            "yields": yield_matrix
        })

    except Exception as e:

        print("Heat map error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500

# ==========================================
# OPEN BROWSER
# ==========================================

def open_browser():

    webbrowser.open_new(
        "http://127.0.0.1:5000/"
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == '__main__':

    Timer(
        1.2,
        open_browser
    ).start()

    app.run(
        port=5000,
        debug=True
    )