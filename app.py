from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models from pickle files
models = {
    'Random Forest': pickle.load(open('models/random_forest_model.pkl', 'rb')),
    'Logistic Regression': pickle.load(open('models/logistic_regression_model.pkl', 'rb')),
    'SVM': pickle.load(open('models/svm_model.pkl', 'rb')),
    'KNN': pickle.load(open('models/knn_model.pkl', 'rb')),
    'Decision Tree': pickle.load(open('models/decision_tree_model.pkl', 'rb'))
}

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        try:
            # Getting the data from the form
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]

            # Prepare the input features for prediction
            features = np.array(features).reshape(1, -1)

            # Predict using each model
            predictions = {model_name: model.predict(features)[0] for model_name, model in models.items()}

            # Calculate accuracy for each model
            accuracies = {model_name: model.score(features, predictions[model_name]) for model_name, model in models.items()}

            return render_template('index.html', predictions=predictions, accuracies=accuracies)
        except KeyError as e:
            # If there's any issue with the form data, return a bad request response
            return f"Error: Missing field {str(e)}. Please fill in all fields.", 400

if __name__ == '__main__':
    app.run(debug=True)
