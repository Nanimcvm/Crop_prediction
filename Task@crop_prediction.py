import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/Nanimcvm/Crop_prediction/refs/heads/main/synthetic_climate_data.csv')

# Sorting data
data = data.sort_values(by='Latitude')

# Handle missing values using SimpleImputer
knn_imputer = KNNImputer(n_neighbors=10, weights='uniform')
data[['Rainfall (mm)', 'Temperature (°C)']] = knn_imputer.fit_transform(data[['Rainfall (mm)', 'Temperature (°C)']])

#Encoding Historical Yield Trend
def encode_category(value):
    if value == 'No Data':  # Check for null values
        return 0
    elif value == 'Low':
        return 1
    elif value == 'Average':
        return 2
    elif value == 'High':
        return 3
    else:
        return -1  # Handle unexpected values

# Apply encoding to the column
data['Historical Yield Trend'] = data['Historical Yield Trend'].apply(encode_category)

#Encoding Crop Suitability
def encode_category_(value):
    if pd.isnull(value):  
        return 0
    elif value == 'Barley':
        return 1
    elif value == 'Cotton':
        return 2
    elif value == 'Groundnut':
        return 3
    elif value == 'Peas':
        return 4
    elif value == 'Wheat':
        return 5
    elif value == 'Maize':
        return 6
    elif value == 'Rice':
        return 7
    elif value == 'None':
        return 8
    else:
        return -1  

# Apply encoding to the column
data['Crop Suitability'] = data['Crop Suitability'].apply(encode_category_)


# Replace 0 with NaN 
data['Crop Suitability'] = data['Crop Suitability'].replace(0, np.nan)
data['Historical Yield Trend'] = data['Historical Yield Trend'].replace(0, np.nan)

# Perform KNN imputation (reshape single column into 2D)
data['Crop Suitability'] = knn_imputer.fit_transform(data[['Crop Suitability']])
data['Historical Yield Trend'] = knn_imputer.fit_transform(data[['Historical Yield Trend']])

# Encode Soil Quality
le_encod = LabelEncoder()
data['Soil Quality'] = le_encod.fit_transform(data['Soil Quality'])


# Convert numerical yield to categorical based on thresholds
def categorize_crop(value):
    if value <= 1:
        return 'Barley'
    elif 1 < value < 3:
        return 'Cotton'
    elif 2 < value < 4:
        return 'Groundnut'
    elif 3 < value < 5:
        return 'Peas'
    elif 4 < value < 6:
        return 'Wheat'
    elif 5 < value < 7:
        return 'Maize'
    elif 6 < value < 8:
        return 'Rice'
    elif 7 < value < 9:
        return 'None'
    else:
        return 'High'

data['Crop Suitability'] = data['Crop Suitability'].apply(categorize_crop)
data['Crop Suitability'] = le_encod.fit_transform(data['Crop Suitability'])

# Convert numerical Crop to categorical based on thresholds
def categorize_yield(value):
    if value == 1:
        return 'Low'
    elif 1 < 3:
        return 'Medium'
    else:
        return 'High'

data['Yield Category'] = data['Historical Yield Trend'].apply(categorize_yield)
data['Yield Category'] = le_encod.fit_transform(data['Yield Category'])

# Define features for all tasks
features = ['Latitude', 'Longitude']

# Task 1: Predict Best Crop to Grow (Classification)
X_crop = data[features]
y_crop = data['Crop Suitability']
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
scaler_crop = StandardScaler()
X_train_crop = scaler_crop.fit_transform(X_train_crop)
X_test_crop = scaler_crop.transform(X_test_crop)
crop_model = KNeighborsClassifier(n_neighbors=500,weights='distance')
crop_model.fit(X_train_crop, y_train_crop)

# Task 2: Predict Potential Yield (Classification)
X_yield = data[features]
y_yield = data['Yield Category']
X_train_yield, X_test_yield, y_train_yield, y_test_yield = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)
scaler_yield = StandardScaler()
X_train_yield = scaler_yield.fit_transform(X_train_yield)
X_test_yield = scaler_yield.transform(X_test_yield)
yield_model = KNeighborsClassifier(n_neighbors=500,weights='uniform')
yield_model.fit(X_train_yield, y_train_yield)

# Task 3: Predict Risk of Crop Failure (Classification)
data['Failure Risk'] = (data['Yield Category'] == le_encod.transform(['Low'])[0]).astype(int)
X_failure = data[features]
y_failure = data['Failure Risk']
X_train_failure, X_test_failure, y_train_failure, y_test_failure = train_test_split(X_failure, y_failure, test_size=0.2, random_state=42)
scaler_failure = StandardScaler()
X_train_failure = scaler_failure.fit_transform(X_train_failure)
X_test_failure = scaler_failure.transform(X_test_failure)
failure_model = KNeighborsClassifier(n_neighbors=550,weights='distance')
failure_model.fit(X_train_failure, y_train_failure)


# Streamlit App
st.title("Climate Data Analysis")
st.header("Insights for Agriculture")


# User Input on Main Page (not in the sidebar)
st.subheader("Enter Input Parameters")

# Input fields on the main page (not in sidebar)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=10.5)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=80.8)


# Creating user input DataFrame
user_input = pd.DataFrame({
    'Latitude': [latitude],
    'Longitude': [longitude]
})

# Scale user input
user_input_scaled_crop = scaler_crop.transform(user_input)
user_input_scaled_yield = scaler_yield.transform(user_input)
user_input_scaled_failure = scaler_failure.transform(user_input)

# Predictions
predicted_crop = crop_model.predict(user_input_scaled_crop)
predicted_crop_label = categorize_crop(predicted_crop)

predicted_yield_category = yield_model.predict(user_input_scaled_yield)
predicted_yield_label = categorize_yield(predicted_yield_category)

predicted_failure_risk = failure_model.predict(user_input_scaled_failure)

# Display Predictions
st.subheader("Predicted Insights")
st.write(f"**Best Crop to Grow**: {predicted_crop_label}")
st.write(f"**Potential Yield Category**: {predicted_yield_label}")
st.write(f"**Risk of Crop Failure**: {'High' if predicted_failure_risk[0] == 1 else 'Low'}")

#visualizations

# Map the categories for visualization only
category_mapping_crop = {
    1: 'Barley',
    2: 'Cotton',
    3: 'Groundnut',
    4: 'Peas',
    5: 'Wheat',
    6: 'Maize',
    7: 'Rice',
    8: 'None'
}
data['Crop_Suitability'] = data['Crop Suitability'].map(category_mapping_crop)

category_mapping_soil = {
    0: 'Low',
    1: 'Medium',
    2: 'High',
}
data['Soil_Quality'] = data['Soil Quality'].map(category_mapping_soil)

category_mapping_yeild = {
    0: 'Low',
    1: 'High',
}
data['Yield_Category'] = data['Yield Category'].map(category_mapping_yeild)

st.subheader("Data Analysis")
#Box plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=data, x='Crop_Suitability', y='Rainfall (mm)', palette='muted')
ax.set_title('Crop Suitability Distribution by Rainfall (mm)')
st.pyplot(fig)

fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.boxplot(data=data, x='Soil_Quality', y='Rainfall (mm)', ax=ax1, palette='muted')
ax1.set_title('Soil Quality Distribution by Rainfall')
st.pyplot(fig1)

#Hist plot
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.histplot(data['Temperature (°C)'], kde=True, bins=30, ax=ax2, color='skyblue')
ax2.set_title('Distribution of Temperature')
st.pyplot(fig2)

#Count plot
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.countplot(data=data, x='Crop_Suitability', order=data['Crop_Suitability'].value_counts().index, ax=ax3, palette='pastel')
ax3.set_title('Count of Crop Suitability Types')
st.pyplot(fig3)

#Violin plot
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.violinplot(data=data, x='Yield_Category', y='Temperature (°C)', ax=ax4, palette='Set2')
ax4.set_title('Temperature Distribution by Historical Yield Trend')
st.pyplot(fig4)

#Line plot
fig5, ax5 = plt.subplots(figsize=(8, 6))
avg_rainfall = data.groupby('Soil Quality')['Rainfall (mm)'].mean().reset_index()
sns.lineplot(data=avg_rainfall, x='Soil Quality', y='Rainfall (mm)', marker='o', ax=ax5, color='red')
ax5.set_title('Average Rainfall by Soil Quality')
st.pyplot(fig5)

fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.histplot(data['Rainfall (mm)'], kde=True, bins=30, ax=ax6, color='green')
ax6.set_title('Distribution of Rainfall')
st.pyplot(fig6)

fig7, ax7 = plt.subplots(figsize=(8, 6))
sns.countplot(data=data, x='Soil_Quality', ax=ax7, palette='pastel')
ax7.set_title('Count of Siol Quality Types')
st.pyplot(fig7)

fig8, ax8 = plt.subplots(figsize=(8, 6))
sns.violinplot(data=data, x='Yield_Category', y='Rainfall (mm)', ax=ax8, palette='Set2')
ax4.set_title('Rainfall Distribution by Historical Yield Trend')
st.pyplot(fig8)

