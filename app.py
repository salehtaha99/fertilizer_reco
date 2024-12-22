from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import pickle

model_filename = "Fertilizersav.sav"
rf_model = pickle.load(open(model_filename, "rb"))

class_Fertilizer = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP',
       'DAP-MOP', 'DAP-SSP', 'DAP-Urea', 'MOP', 'MOP-SSP', 'MOP-Urea',
       'SSP', 'SSP-DAP', 'SSP-DAP-Urea', 'SSP-MOP', 'SSP-Urea', 'Urea',
       'Urea-DAP', 'Urea-DAP-MOP', 'Urea-SSP', 'Urea-SSP-MOP']  

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <html>
        <head>
            <title>Fertilizer Recommendation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }
                h1 {
                    color: #2c3e50;
                }
                form {
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                    width: 300px;
                }
                label {
                    color: #2c3e50;
                }
                input, button {
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    outline: none;
                }
                button {
                    background-color: #2c3e50;
                    color: #ffffff;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #34495e;
                }
                .result {
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #e7f7e7;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    color: #2c7d2e;
                    font-size: 1.2em;
                }
            </style>
        </head>
        <body>
            <h1>Fertilizer Recommendation System</h1>
            <form action="/predict" method="post">
                <label>Soil Type (String):</label>
                <input type="text" name="soil_type" required><br><br>
                <label>Crop Type (String):</label>
                <input type="text" name="crop_type" required><br><br>
                <label>N (Nitrogen):</label>
                <input type="number" name="n" step="1" required><br><br>
                <label>P (Phosphorus):</label>
                <input type="number" name="p" step="1" required><br><br>
                <label>K (Potassium):</label>
                <input type="number" name="k" step="1" required><br><br>
                <button type="submit">Recommend Fertilizer</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict(request: Request):
    # Check the Content-Type of the request
    if request.headers.get("Content-Type") == "application/json":
        data = await request.json()
        soil_type = data.get("soil_type", "").lower()
        crop_type = data.get("crop_type", "").lower()
        n = data.get("n")
        p = data.get("p")
        k = data.get("k")
    else:
        form_data = await request.form()
        soil_type = form_data.get("soil_type", "").lower()
        crop_type = form_data.get("crop_type", "").lower()
        n = float(form_data.get("n"))
        p = float(form_data.get("p"))
        k = float(form_data.get("k"))

    soil_type_dict = {'black': 0, 'clayey': 1, 'loamy': 2, 'red': 3, 'sandy': 4}
    crop_type_dict = {'barley': 0, 'cotton': 1, 'cucumber': 2, 'maize': 3, 'millets': 4, 'oil seeds': 5, 'paddy': 6, 'sugarcane': 7, 'tobacco': 8, 'wheat': 9, 'rice': 10, 'tomatoes': 11}

    soil_type_encoded = soil_type_dict.get(soil_type, -1)
    crop_type_encoded = crop_type_dict.get(crop_type, -1)

    if soil_type_encoded == -1 or crop_type_encoded == -1:
        return JSONResponse({"error": "Invalid soil type or crop type. Please use valid values."}, status_code=422)

    features = np.array([[soil_type_encoded, crop_type_encoded, n, p, k]])
    recommended_fertilizer_index = rf_model.predict(features)[0]
    recommended_fertilizer = class_Fertilizer[recommended_fertilizer_index]

    if request.headers.get("Content-Type") == "application/json":
        return JSONResponse({
            "recommended_fertilizer": recommended_fertilizer,
            "soil_type": soil_type,
            "crop_type": crop_type,
            "n": n,
            "p": p,
            "k": k
        })

    result_html = f"""
    <html>
        <head>
            <title>Fertilizer Recommendation Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }}
                .result {{
                    padding: 20px;
                    background-color: #e7f7e7;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    color: #2c7d2e;
                    font-size: 1.5em;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="result">
                <h2>Recommended Fertilizer: {recommended_fertilizer}</h2>
                <p>Soil Type: {soil_type}</p>
                <p>Crop Type: {crop_type}</p>
                <p>N (Nitrogen): {n}</p>
                <p>P (Phosphorus): {p}</p>
                <p>K (Potassium): {k}</p>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=result_html)
