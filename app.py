from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))


# Route for the home page
@app.route('/')
def home():
    return render_template('form.html')


# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    feature1 = request.form['feature1']
    feature2 = request.form['feature2']
    feature3 = request.form['feature3']
    feature4 = request.form['feature4']
    feature5 = request.form['feature5']
    feature6 = request.form['feature6']
    feature7 = request.form['feature7']
    feature8 = request.form['feature8']

    # Convert features to a NumPy array (assuming numerical input)
    features = np.array([[float(feature1), float(feature2), float(feature3), float(feature4), float(feature5), float(feature6), float(feature7), float(feature8)]])

    input_data_replicated = np.tile(features, (20, 1))  # Now it's (20, 8)

    # Reshape to (1, 20, 8) for model prediction
    input_data_replicated = input_data_replicated.reshape(1, 20, 8)

    prediction = round(model.predict(input_data_replicated)[0][0]*100,2)

    # Pass the inputs and the prediction to the result page
    return render_template('form.html', prediction=prediction, feature1=feature1, feature2=feature2, feature3=feature3, feature4=feature4, feature5=feature5, feature6=feature6, feature7=feature7, feature8=feature8)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
