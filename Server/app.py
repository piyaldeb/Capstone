import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import csv


from sklearn.metrics import accuracy_score

# Load your pre-trained sonar dataset (replace with your data loading logic)

# Replace 'sonar_data.csv' with the actual path to your CSV file
with open('Server/Copy of sonar data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    sonar_data = list(reader)  # Create a list of lists from the CSV data
# Feature selection (drop last column)

# Convert the list of lists to a DataFrame
sonar_data = pd.DataFrame(sonar_data)

# Feature selection (drop last column)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Train-test split (consider using a larger test set for better evaluation)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)


# Function to make predictions
def predict_mine_or_rock(data):
    data_array = np.asarray(data).reshape(1, -1)  # Reshape for single prediction
    prediction = model.predict(data_array)[0]
    return "Rock" if prediction == 'R' else "Mine"


# Flask app setup
from flask import Flask, render_template, request

# app = Flask(__name__)
app=Flask(__name__,template_folder='../templates')






  # Replace with the actual path
 # Replace with the actual path

@app.route("/")
def index():
    return render_template("index.html")  # Render the HTML template


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form["data"]  # Get data from the form
    data_list = [float(x) for x in data.split(",")]  # Convert comma-separated string to list of floats
    prediction = predict_mine_or_rock(data_list)
    return render_template("result.html", prediction=prediction)

@app.route("/about")
def about():
    # Replace with your desired content about the model or data
    about_text = """
    This is a basic sonar data classification model that uses Logistic Regression to distinguish between rocks and mines based on sonar readings.

    You can enter comma-separated sonar data in the home page to get a prediction.

    This model is for demonstration purposes only and may not be accurate for real-world applications.
    """
    return render_template("about.html", about_text=about_text)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010, debug=True)
 # Run the Flask app in debug mode for development
