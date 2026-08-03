# 📱 Social Media Addiction Prediction Using Machine Learning

## Overview

This project is a Machine Learning-based web application developed using Flask. It predicts a user's social media addiction level based on daily usage habits such as screen time, app opens, sleep hours, study hours, and mood level.

The application uses the Logistic Regression algorithm for classification and provides a user-friendly interface for entering data, viewing predictions, and visualizing the results through a bar chart.

---

## Features

- Predicts Social Media Addiction Level
- Machine Learning using Logistic Regression
- Data preprocessing using StandardScaler
- Interactive Flask web application
- Dynamic result page with color themes
- Graphical visualization using Matplotlib
- Responsive HTML and CSS interface

---

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Matplotlib
- HTML5
- CSS3

---

## Project Structure

```
Social-Media-Addiction-Prediction/
│
├── app.py
├── ml.py
├── social_media_addiction_dataset.csv
├── requirements.txt
├── README.md
│
├── templates/
│   ├── welcome.html
│   ├── index.html
│   └── result.html
│
└── static/
    ├── style.css
    └── graph.png
```

---

## Dataset Features

- Screen Time
- App Opens
- Sleep Hours
- Study Hours
- Mood Level

### Target Variable

- Low
- Medium
- High

---

## Machine Learning Model

The project uses the **Logistic Regression** algorithm for classification.

### Python Files

### app.py
- Main Flask application
- Handles routing and user interaction
- Integrates the Machine Learning model
- Displays prediction results and graphs

### ml.py
- Standalone Machine Learning implementation
- Demonstrates Logistic Regression model training and prediction
- Included to show the ML workflow used during development

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/Social-Media-Addiction-Prediction.git
```

Move into the project folder

```bash
cd Social-Media-Addiction-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## Future Enhancements

- Train the model with a larger dataset
- Improve prediction accuracy
- Store user prediction history
- Deploy the application online
- Add user authentication

---

## Limitations

- Uses a small dataset for demonstration.
- Prediction accuracy depends on the dataset quality.
- Intended for educational purposes.

---


Developed as a Machine Learning project using Flask and Logistic Regression.
