from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model safely
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
        return render_template('index.html', prediction_text="Error: Model file could not be loaded on the server.")
    
    try:
        form_keys = [
            'Education', 'JoiningYear', 'City', 'PaymentTier', 
            'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
        ]
        
        float_features = []
        for key in form_keys:
            val = request.form.get(key)
            if val is None or val.strip() == '':
                raise ValueError(f"Missing value for required field: {key}")
            float_features.append(float(val))
        
        final_features = np.array([float_features])
        prediction = model.predict(final_features)
        output = int(prediction[0])
        
        status = "Likely to Leave (Churn)" if output == 1 else "Likely to Stay"
        result_msg = f"Prediction Result: {status} (Class Code: {output})"
        
        return render_template('index.html', prediction_text=result_msg)
        
    except Exception as e:
        return render_template('index.html', prediction_text=f'Processing Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
