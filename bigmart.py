import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------
# Load Train Dataset
# -----------------------------------

train_df = pd.read_csv("Train.csv")

print("First 5 Rows:\n")
print(train_df.head())

print("\nDataset Info:\n")
print(train_df.info())

# -----------------------------------
# Handle Missing Values
# -----------------------------------

train_df['Item_Weight'] = train_df['Item_Weight'].fillna(
    train_df['Item_Weight'].mean()
)

train_df['Outlet_Size'] = train_df['Outlet_Size'].fillna(
    train_df['Outlet_Size'].mode()[0]
)

# -----------------------------------
# Clean Data
# -----------------------------------

train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

# -----------------------------------
# Encode Categorical Columns
# -----------------------------------

le = LabelEncoder()

categorical_cols = [
    'Item_Fat_Content',
    'Item_Type',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type'
]

for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])

# -----------------------------------
# Features and Target
# -----------------------------------

X = train_df.drop([
    'Item_Outlet_Sales',
    'Item_Identifier',
    'Outlet_Identifier'
], axis=1)

y = train_df['Item_Outlet_Sales']

# -----------------------------------
# Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# Train Random Forest Model
# -----------------------------------

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

# -----------------------------------
# Predictions
# -----------------------------------

y_pred = rf.predict(X_test)

print("\n----- MODEL PERFORMANCE -----")

print("MAE:",
      mean_absolute_error(y_test, y_pred))

print("R2 Score:",
      r2_score(y_test, y_pred))

# -----------------------------------
# Actual vs Predicted Graph
# -----------------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# -----------------------------------
# Feature Importance
# -----------------------------------

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:\n")
print(importance_df)

# -----------------------------------
# Load Test Dataset
# -----------------------------------

test_df = pd.read_csv("Test(1).csv")

# -----------------------------------
# Handle Missing Values
# -----------------------------------

test_df['Item_Weight'] = test_df['Item_Weight'].fillna(
    test_df['Item_Weight'].mean()
)

test_df['Outlet_Size'] = test_df['Outlet_Size'].fillna(
    test_df['Outlet_Size'].mode()[0]
)

# -----------------------------------
# Clean Data
# -----------------------------------

test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

# -----------------------------------
# Encode Categorical Columns
# -----------------------------------

for col in categorical_cols:
    test_df[col] = le.fit_transform(test_df[col])

# -----------------------------------
# Prepare Test Features
# -----------------------------------

X_final = test_df.drop([
    'Item_Identifier',
    'Outlet_Identifier'
], axis=1)

# -----------------------------------
# Scale Test Data
# -----------------------------------

X_final = scaler.transform(X_final)

# -----------------------------------
# Final Predictions
# -----------------------------------

final_predictions = rf.predict(X_final)

# -----------------------------------
# Save Predictions
# -----------------------------------

output = pd.DataFrame({
    'Item_Identifier': test_df['Item_Identifier'],
    'Outlet_Identifier': test_df['Outlet_Identifier'],
    'Predicted_Sales': final_predictions
})

output.to_csv("Predicted_Output.csv", index=False)

print("\nPredictions saved in Predicted_Output.csv")

# -----------------------------------
# Sample Prediction
# -----------------------------------

print("\nFirst 10 Predictions:\n")
print(output.head(10))