from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        Pclass = int(request.form['Pclass'])
        Sex = int(request.form['Sex'])
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = int(request.form['Embarked'])
        FamilySize = SibSp + Parch + 1
        Title = int(request.form['Title'])

        features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, Title]])

        prediction = model.predict(features)

        result = "Survived" if prediction[0] == 1 else "Did not Survive"

        return render_template('index.html', prediction_text=f"The passenger would have: {result}")

if __name__ == "__main__":
    app.run(debug=True)
