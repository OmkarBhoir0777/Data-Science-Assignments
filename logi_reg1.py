# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:46:05 2024

@author: Omkar
"""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
st.title('Logistic Regression on Titanic Dataset')

st.header('1. Data Exploration')
train_data_path = 'Titanic_train.csv'
test_data_path = 'Titanic_test.csv'

@st.cache_data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

train_data, test_data = load_data(train_data_path, test_data_path)
st.subheader('Train Data')
st.write(train_data.head())

# Data summary
st.write(train_data.describe())

# Visualizations
st.subheader('Visualizations')

# Histograms
fig, ax = plt.subplots(figsize=(10, 8))
train_data.hist(bins=20, ax=ax)
plt.tight_layout()
st.pyplot(fig)

# Pairplot
st.write('Pairplot')
pairplot_fig = sns.pairplot(train_data, hue='Survived')
st.pyplot(pairplot_fig)

# Data Preprocessing
st.header('2. Data Preprocessing')

# Handle missing values
st.subheader('Handling Missing Values')
st.write('Number of missing values in each column:')
st.write(train_data.isnull().sum())

# Fill missing values
train_data = train_data.assign(
    Age=lambda df: df['Age'].fillna(df['Age'].median()),
    Embarked=lambda df: df['Embarked'].fillna(df['Embarked'].mode()[0])
)
test_data = test_data.assign(
    Age=lambda df: df['Age'].fillna(df['Age'].median()),
    Fare=lambda df: df['Fare'].fillna(df['Fare'].median())
)

# Encode categorical variables
st.subheader('Encoding Categorical Variables')

# Combine the datasets to ensure consistent encoding
combined_data = pd.concat([train_data, test_data]).reset_index(drop=True)

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

combined_data['Sex'] = le_sex.fit_transform(combined_data['Sex'])
combined_data['Embarked'] = le_embarked.fit_transform(combined_data['Embarked'])

# Split the combined data back into train and test sets
train_data['Sex'] = combined_data.loc[train_data.index, 'Sex'].values
train_data['Embarked'] = combined_data.loc[train_data.index, 'Embarked'].values
test_data['Sex'] = combined_data.loc[test_data.index, 'Sex'].values
test_data['Embarked'] = combined_data.loc[test_data.index, 'Embarked'].values

st.write('Data after encoding:')
st.write(train_data.head())

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']

# Model Building
st.header('3. Model Building')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

st.write('Model Trained Successfully!')

# Model Evaluation
st.header('4. Model Evaluation')

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))

# Confusion matrix
st.write('Confusion Matrix:')
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

st.write('ROC Curve:')
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

# Interpretation
st.header('5. Interpretation')
st.write('Logistic Regression Coefficients:')
coeff_df = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])
st.write(coeff_df)

st.write('Significance of Features:')
st.write("Interpretation based on coefficients: Higher coefficient means a stronger influence on the prediction.")

# Deployment with Streamlit
st.header('6. Deployment')

st.subheader('Make Predictions')
user_input = {}
for feature in features:
    if feature == 'Sex':
        # Provide user with option to select original categories
        user_input[feature] = st.selectbox(f'Select {feature}:', le_sex.classes_)
    elif feature == 'Embarked':
        user_input[feature] = st.selectbox(f'Select {feature}:', le_embarked.classes_)
    else:
        user_input[feature] = st.number_input(f'Enter {feature}:', value=float(train_data[feature].median()))

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Encode user input using the same encoders
user_df['Sex'] = le_sex.transform(user_df['Sex'])
user_df['Embarked'] = le_embarked.transform(user_df['Embarked'])

# Standardize user input
user_df_scaled = scaler.transform(user_df)

# Predict
prediction = model.predict(user_df_scaled)
prediction_prob = model.predict_proba(user_df_scaled)[0][1]

st.write(f'Predicted Survival: {"Survived" if prediction[0] == 1 else "Not Survived"}')
st.write(f'Survival Probability: {prediction_prob:.2f}')
