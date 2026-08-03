from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)


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

    # Create input data
    new_data = pd.DataFrame(
        [[screen_time,
          app_opens,
          sleep_hours,
          study_hours,
          mood_level]],
        columns=[
            "Screen_Time",
            "App_Opens",
            "Sleep_Hours",
            "Study_Hours",
            "Mood_Level"
        ]
    )

    # Predict
    prediction = model.predict(new_data)[0]

    levels = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    result = levels[prediction]

    # Color
    if result == "Low":
        css_class = "low"
    elif result == "Medium":
        css_class = "medium"
    else:
        css_class = "high"

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

    plt.bar(labels, values)

    plt.title("Your Social Media Usage Profile")
    plt.ylabel("Value")
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