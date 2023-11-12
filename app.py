import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('C:/Users/rana/Downloads/Updated_Data.csv')

# Separate features (X) and target variable (y)
X = df.drop('Credit knowledge', axis=1)
y = df['Credit knowledge']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Apply the model to the entire dataset
predictions = model.predict(X)

# Add the predictions to the original DataFrame
df['Predicted_Credit_knowledge'] = predictions

# Save the updated DataFrame to a new CSV file
df.to_csv('output_dataset.csv', index=False)



import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib  # Use joblib for model persistence

# Load the pre-trained logistic regression model
# try:
#     model = joblib.load('C:/Users/rana/OneDrive - UNT System/Desktop/Hackathon/Logistic_Regression.py')  # Adjust the file name if needed
#     print("Model loaded successfully.")
# except FileNotFoundError:
    #print("No pre-trained model found. Please train the model first.")
    # If the model is not trained, you need to train it first
    # X_train, X_test, y_train, y_test should be defined and used for training
    # model.fit(X_train, y_train)
    # joblib.dump(model, 'logistic_model.pkl')
    # print("Model trained and saved.")

# Get the path of the new CSV file from the user
#new_file_path = input("Enter the path of the new CSV file: ")

# Load the new dataset
new_df = pd.read_csv('C:/Users/rana/Downloads/Book1.csv')

# Separate features (X)
X_new = new_df.drop('Credit knowledge', axis=1)

# Apply the pre-trained logistic regression model to make predictions
predictions_new = model.predict(X_new)

# Add the predictions to the new DataFrame
new_df['Credit knowledge'] = predictions_new

# Save the updated DataFrame to a new CSV file
new_file_name = 'output_new_dataset.csv'
new_df.to_csv(new_file_name, index=False)

print(f"Updated DataFrame saved to {new_file_name}")


# Calculate the percentage of 1's and 0's in the 'Credit knowledge' column
percentage_ones = (new_df['Credit knowledge'].sum() / len(new_df)) * 100
percentage_zeros = 100 - percentage_ones

print(f"Percentage of 1's: {percentage_ones:.2f}%")
print(f"Percentage of 0's: {percentage_zeros:.2f}%")




from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load the pre-trained logistic regression model
#model = joblib.load('logistic_model.pkl')  # Adjust the file name if needed

html_template = """
<!DOCTYPE html>
<html>

<head>
    <title>My App</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
        integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
</head>

<body class="d-flex align-items-center justify-content-center"
    style="min-height: 100vh; background-image: url('../../assets/Fidelity_Investments_4823505.jpg'); background-size: cover;">

    <div class="container">
        <div class="row">
            <div class="col-md-6">
                <form action="/api/predict" method="post">
                    <label>Select State:</label>
                    <select name="state" class="form-control">
                        <option value="state1">State 1</option>
                        <option value="state2">State 2</option>
                        <option value="state3">State 3</option>
                        <option value="state4">State 4</option>
                        <option value="state5">State 5</option>
                    </select>
            </div>
            <div class="col-md-6">
                <label>Upload CSV File:</label>
                <input type="file" name="file" class="form-control" />
            </div>
        </div>

        <div class="row mt-3">
            <div class="col-md-6">
                <button class="btn btn-primary" type="submit">Submit</button>
                </form>
            </div>
        </div>

        <div class="row mt-3">
            <div class="col-md-12">
                <div id="result"></div>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"
        integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo"
        crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"
        integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1"
        crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"
        integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM"
        crossorigin="anonymous"></script>

    <script>
        // Handle form submission
        document.querySelector('form').addEventListener('submit', function (event) {
            event.preventDefault();

            // Display loading message
            document.getElementById('result').innerHTML = 'Loading...';

            // Perform form submission via AJAX
            var formData = new FormData(this);

            fetch('/api/predict', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    // Display the results
                    document.getElementById('result').innerHTML = `
                        <p>Percentage of 1's: ${data.percentage_ones.toFixed(2)}%</p>
                        <p>Percentage of 0's: ${data.percentage_zeros.toFixed(2)}%</p>
                    `;
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('result').innerHTML = 'Error occurred.';
                });
        });
    </script>
</body>

</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the request
        state = request.form.get('state')
        uploaded_file = request.files['file']

        # Process the file (assuming it's a CSV file)
        df = pd.read_csv(uploaded_file)
        # Additional processing based on your specific requirements

        # Apply the pre-trained logistic regression model to make predictions
        predictions = model.predict(df.drop('Credit knowledge', axis=1))

        # Calculate the percentage of 1's and 0's in the predictions
        percentage_ones = (predictions.sum() / len(predictions)) * 100
        percentage_zeros = 100 - percentage_ones

        # Return the results as JSON
        return jsonify({
            'People with Credit Knowledge': percentage_ones,
            'People without Credit Knowledge': percentage_zeros
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

