**Crop Prediction and Analysis Application**

This project is a machine learning-based application designed to provide insights for agriculture using climate and soil data. The application predicts the following:

Best Crop to Grow based on location.
Potential Yield Category for the crop.
Risk of Crop Failure based on yield trends.

The application is built using Python, with Streamlit for the user interface and Scikit-learn for machine learning.

Features

1. Best Crop to Grow-Predicts the most suitable crop to grow based on the latitude and longitude.

2. Potential Yield Category-Classifies the yield potential as "Low", "Medium", or "High" based on historical trends and location data.

3. Risk of Crop Failure-Assesses the risk of crop failure by analyzing yield trends and predicting failure probability.


Dataset

The application uses synthetic climate data sourced from the following URL:

https://raw.githubusercontent.com/Nanimcvm/Crop_prediction/refs/heads/main/synthetic_climate_data.csv


Key Dataset Columns:

Latitude and Longitude: Location coordinates.
Rainfall (mm) and Temperature (°C): Climate metrics.
Crop Suitability: Target column for best crop prediction.
Historical Yield Trend: Indicates past yield trends (Low, Average, High).
Soil Quality: Soil condition (encoded).


Requirements

Install the required Python packages:
pip install pandas numpy scikit-learn streamlit matplotlib seaborn


Code Overview

1. Preprocessing

Missing Values:Used KNN Imputer to fill missing values in Rainfall (mm) and Temperature (°C).
Encoded categorical features into numeric values.

Feature Scaling:Used StandardScaler to standardize features for model training.

2. Models

Built KNN classifiers for each task using the KNeighborsClassifier from Scikit-learn.
Best Crop Prediction: Classifies the most suitable crop based on location.
Yield Prediction: Categorizes the yield trend into Low, Medium, or High.
Failure Risk Prediction: Predicts the likelihood of crop failure.

3. Streamlit Application

Interactive user interface with the following inputs:
Latitude(-90 to 90)
Longitude(-180 to 180)

Displays predictions for:
Best crop to grow.
Potential yield category.
Risk of crop failure.


How to Run
Clone the repository and navigate to the project directory.

Run the Streamlit application:
streamlit run <script_name>.py
Enter the latitude and longitude in the input fields.
View the predictions and model accuracy metrics.

Model Performance
The application calculates accuracy scores for each task and displays them on the Streamlit interface:

Crop Prediction Model Accuracy - 40.10

Yield Prediction Model Accuracy - 75.30

Failure Risk Prediction Model Accuracy - 75.20 


Customization

Adjust KNN Parameters: Modify the n_neighbors and weights parameters in the KNeighborsClassifier to optimize performance.
Add Features: Include more features such as soil pH, elevation, or other relevant agricultural data.
Enhance Visualization: Use Matplotlib and Seaborn for more detailed visual insights.


Dependencies

Python 
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn