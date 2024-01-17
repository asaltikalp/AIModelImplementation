import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

data = pd.read_csv('C:\Users\agask\OneDrive\Resimler\Belgeler\AIIndividualProject\hwDataSet.xlsx')

label_encoders = {}
categorical_columns = ['Brand', 'Brand_First_Char']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

def feature_engineering(data):
    data['combined_feature'] = data.iloc[:, 0] / (data.iloc[:, 1] +
0.001)
    data['log_transformed_feature'] = np.log(data.iloc[:, 2] + 1)
    return data

data_fe = feature_engineering(data.copy())

X = data_fe.drop(['ResponseCode'], axis=1)
y = data_fe['ResponseCode']

X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=42)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

rf_cross_val_scores = cross_val_score(rf_classifier, X, y, cv=5)

print("Cross-Validation Scores for Random Forest: ",rf_cross_val_scores)
print("Average Score: ", rf_cross_val_scores.mean())
print("Standard Deviation:", rf_cross_val_scores.std())
