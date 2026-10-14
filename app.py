from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Naive Bayes model with its proper filename
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in the exact order expected by the model[cite: 1]
        education = float(request.form['Education'])
        joining_year = float(request.form['JoiningYear'])
        city = float(request.form['City'])
        payment_tier = float(request.form['PaymentTier'])
        age = float(request.form['Age'])
        gender = float(request.form['Gender'])
        ever_benched = float(request.form['EverBenched'])
        experience = float(request.form['ExperienceInCurrentDomain'])
        
        # Combine into array
        final_features = np.array([[
            education, joining_year, city, payment_tier, 
            age, gender, ever_benched, experience
        ]])
        
        # Make prediction
        prediction = model.predict(final_features)
        output = prediction[0]
        
        result_msg = f"Prediction Result: {output} (1 = Likely to Leave, 0 = Likely to Stay)"
        return render_template('index.html', prediction_text=result_msg)
        
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in processing: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
