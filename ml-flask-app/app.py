from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('student_model.pkl')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        attendance = float(request.form["attendance"])
        mid_marks = float(request.form["mid_marks"])

        # Scale mid marks (out of 30 → scale to 100)
        mid_scaled = mid_marks * (100 / 30)

        # Make prediction
        input_df = pd.DataFrame([[attendance, mid_marks]],
                                columns=["Attendance(%)", "Mid_Marks"])
        prediction = model.predict(input_df)[0]

        return render_template("index.html",
                               prediction_text=f"Predicted CGPA: {round(float(prediction), 2)}")
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
