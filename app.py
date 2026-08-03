from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("social_media_addiction_dataset.csv")

# Convert labels to numbers
df["Addiction_Level"] = df["Addiction_Level"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

# Features and Target
X = df[[
    "Screen_Time",
    "App_Opens",
    "Sleep_Hours",
    "Study_Hours",
    "Mood_Level"
]]
y = df["Addiction_Level"]

# Scale Features to balance weights
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model on Scaled Data
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/form")
def form():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    screen_time = float(request.form["screen_time"])
    app_opens = float(request.form["app_opens"])
    sleep_hours = float(request.form["sleep_hours"])
    study_hours = float(request.form["study_hours"])
    mood_level = float(request.form["mood_level"])

    # Create input data DataFrame
    new_data = pd.DataFrame([[
        screen_time,
        app_opens,
        sleep_hours,
        study_hours,
        mood_level
    ]], columns=[
        "Screen_Time",
        "App_Opens",
        "Sleep_Hours",
        "Study_Hours",
        "Mood_Level"
    ])

    # Scale the user's input values using the training scaler
    new_data_scaled = scaler.transform(new_data)

    # Machine Learning Prediction
    prediction = model.predict(new_data_scaled)[0]

    levels = {
        0: "Low",
        1: "Medium",
        2: "High"
    }
    result = levels[prediction]

    # --- Pure Screen Time Guard Filters (No App Opens) ---
    # Low Condition: 2 hours or less of Screen Time
    if screen_time <= 2:
        result = "Low"
        
    # High Condition: 7 hours or more of Screen Time
    elif screen_time >= 7:
        result = "High"
        
    # Medium Condition: Between 3 and 6 hours of Screen Time
    else:
        result = "Medium"

    # --- Setup CSS Classes and Graph Colors based on the final scale results ---
    if result == "High":
        css_class = "high"
        bar_color = "#e74c3c"  # Red
    elif result == "Medium":
        css_class = "medium"
        bar_color = "#f39c12"  # Orange
    else:
        css_class = "low"
        bar_color = "#2ecc71"  # Green

    # Create Graph
    plt.figure(figsize=(8, 5))

    labels = [
        "Screen Time",
        "App Opens",
        "Sleep Hours",
        "Study Hours",
        "Mood Level"
    ]

    values = [
        screen_time,
        app_opens,
        sleep_hours,
        study_hours,
        mood_level
    ]

    # Dynamically match chart bars to the user's profile result color scheme
    plt.bar(labels, values, color=bar_color)
    plt.title("Your Social Media Usage Profile")
    plt.ylabel("Values")
    plt.tight_layout()

    plt.savefig("static/graph.png")
    plt.close()

    return render_template(
        "result.html",
        result=result,
        css_class=css_class
    )


if __name__ == "__main__":
    app.run(debug=True)
