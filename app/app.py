from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model_path = 'model/model.pkl'  # or the correct relative path inside the container
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tv = float(request.form['tv'])
        prediction = model.predict(np.array([[tv]]))[0]
        return render_template('index.html', prediction_text=f"Predicted Sales: {prediction:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
