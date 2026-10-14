from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Naive Bayes model safely
try:
    with open('naive_bayes_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model could not be loaded on the server.")
    
    try:
        # Extract features safely with error checking
        form_keys = [
            'Education', 'JoiningYear', 'City', 'PaymentTier', 
            'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
        ]
        
        float_features = []
        for key in form_keys:
            val = request.form.get(key)
            if val is None or val.strip() == '':
                raise ValueError(f"Missing value for field: {key}")
            float_features.append(float(val))
        
        # Reshape into a 2D array for the model
        final_features = np.array([float_features])
        
        # Make prediction
        prediction = model.predict(final_features)
        output = prediction[0]
        
        result_msg = f"Prediction Result: {output} (1 = Likely to Leave, 0 = Likely to Stay)"
        return render_template('index.html', prediction_text=result_msg)
        
    except Exception as e:
        # This will catch input mismatch or conversion errors and display them cleanly
        return render_template('index.html', prediction_text=f'Processing Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
